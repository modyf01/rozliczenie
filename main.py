from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import re
from bs4 import BeautifulSoup
from datetime import timedelta, datetime

app = Flask(__name__)

# Globalny zbiór transakcji oraz licznik unikalnych identyfikatorów
all_trades_df = pd.DataFrame(
    columns=["id", "waluty", "Stock", "Date/Time", "Quantity", "Proceeds", "Comm/Fee", "Basis"]
)
next_transaction_id = 1


# ----------------- Funkcje pomocnicze -----------------
def parse_html_transactions(html_content: str) -> pd.DataFrame:
    """
    Parsuje HTML i zwraca DataFrame z danymi transakcji.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    parent_container = soup.find(
        lambda tag: tag.name == "div" and tag.get("id") and re.search(r"^tblTransactions_.*Body$", tag.get("id"))
    )
    if parent_container is None:
        raise ValueError("Nie znaleziono kontenera z transakcjami")
    trades_table = parent_container.find("table")
    if trades_table is None:
        raise ValueError("Nie znaleziono tabeli z transakcjami")

    data = []
    current_currency = None
    for tr in trades_table.find_all("tr"):
        cells = tr.find_all("td")
        # Wiersze z jedną komórką traktujemy jako nagłówek waluty
        if len(cells) == 1:
            text = cells[0].get_text(strip=True)
            if text in ["EUR", "GBP", "USD", "PLN"]:
                current_currency = text
            continue
        if len(cells) >= 8:
            stock = cells[0].get_text(strip=True)
            date_time = cells[1].get_text(strip=True)
            quantity = cells[2].get_text(strip=True)
            proceeds = cells[5].get_text(strip=True)
            comm_fee = cells[6].get_text(strip=True)
            basis = cells[7].get_text(strip=True)
            data.append({
                "waluty": current_currency,
                "Stock": stock,
                "Date/Time": date_time,
                "Quantity": quantity,
                "Proceeds": proceeds,
                "Comm/Fee": comm_fee,
                "Basis": basis
            })
    return pd.DataFrame(data)


def filter_and_convert_transactions(df_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Filtrowanie wierszy oraz konwersja kolumn liczbowych i dat.
    """
    df_trades = df_trades[~((df_trades["Quantity"].str.strip() == "") &
                             (df_trades["Stock"].str.lower().str.contains("total")))]
    df_trades = df_trades[df_trades["Basis"].str.strip() != ""]

    for col in ["Quantity", "Proceeds", "Comm/Fee", "Basis"]:
        df_trades[col] = pd.to_numeric(df_trades[col].str.replace(",", ""), errors="coerce")
    df_trades["Date/Time"] = pd.to_datetime(df_trades["Date/Time"], errors="coerce")
    return df_trades


def load_exchange_rates(csv_path: str) -> pd.DataFrame:
    """
    Ładuje kursy walut z pliku CSV i formatuje daty.
    """
    df_kursy = pd.read_csv(csv_path, delimiter=",")
    df_kursy["data"] = pd.to_datetime(df_kursy["data"], format="%Y%m%d", errors="coerce")
    return df_kursy.sort_values("data")


def merge_exchange_rates(df_trades: pd.DataFrame, df_kursy: pd.DataFrame) -> pd.DataFrame:
    """
    Łączy transakcje z kursami walut, wykorzystując datę transakcji pomniejszoną o jeden dzień.
    """
    df_trades["match_date"] = df_trades["Date/Time"] - timedelta(days=1)
    df_trades = df_trades.sort_values("match_date")
    df_trades = pd.merge_asof(df_trades, df_kursy, left_on="match_date", right_on="data", direction="backward")
    df_trades.rename(columns={"data": "Kurs_Date"}, inplace=True)
    df_trades.drop(columns=["match_date"], inplace=True)
    return df_trades


def apply_currency_conversion(df_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Przelicza wartości transakcji zgodnie z odpowiednim kursem waluty.
    """

    def wybierz_kurs(row):
        waluta = row["waluty"]
        if waluta == "USD":
            return row.get("1 USD")
        elif waluta == "EUR":
            return row.get("1 EUR")
        elif waluta == "GBP":
            return row.get("1 GBP")
        elif waluta == "PLN":
            return 1.0
        return None

    df_trades["rate"] = df_trades.apply(wybierz_kurs, axis=1)
    df_trades["Basis_converted"] = df_trades["Basis"] * df_trades["rate"]
    df_trades["Comm/Fee_converted"] = df_trades["Comm/Fee"] * df_trades["rate"]
    df_trades["Proceeds_converted"] = df_trades["Proceeds"] * df_trades["rate"]
    df_trades = df_trades.drop(columns=["1 USD", "1 EUR", "1 GBP"], errors="ignore")

    desired_order = ["id", "waluty", "Stock", "Date/Time", "Quantity", "Proceeds", "Proceeds_converted",
                     "Comm/Fee", "Basis", "Kurs_Date", "rate", "Basis_converted", "Comm/Fee_converted"]
    return df_trades[desired_order]


def allocate_fifo(df_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Alokacja FIFO – przypisuje transakcjom kupna wykorzystanie zgodnie z kolejnością transakcji.
    """
    df_trades["fifo_allocated"] = 0.0
    df_trades["fifo_used"] = False

    # Sprzedaże są zawsze w całości wykorzystane
    df_trades.loc[df_trades["Quantity"] < 0, "fifo_used"] = True

    for stock, group in df_trades.groupby("Stock"):
        group = group.sort_values("Date/Time").copy()
        sells = group[group["Quantity"] < 0]
        total_sold = -sells["Quantity"].sum() if not sells.empty else 0.0

        # Transakcje kupna, które nie są jeszcze wykorzystane
        buys = group[(group["Quantity"] > 0) & (~group["fifo_used"])].sort_values("Date/Time").copy()
        remaining = total_sold
        for idx, buy in buys.iterrows():
            if remaining <= 0:
                break
            available = buy["Quantity"]
            if available <= remaining:
                df_trades.at[idx, "fifo_allocated"] = available
                df_trades.at[idx, "fifo_used"] = True
                remaining -= available
            else:
                df_trades.at[idx, "fifo_allocated"] = remaining
                df_trades.at[idx, "fifo_used"] = True
                remaining = 0
    return df_trades


def summarize_transactions(group: pd.DataFrame) -> pd.DataFrame:
    """
    Generuje podsumowanie transakcji dla danego symbolu akcji.
    """
    total_sold_sum = 0.0
    proceeds_sum = 0.0
    proceeds_conv_sum = 0.0
    comm_fee_sum = 0.0
    basis_sum = 0.0
    basis_conv_sum = 0.0
    comm_fee_conv_sum = 0.0

    for idx, row in group.iterrows():
        if row["fifo_used"]:
            if row["Quantity"] < 0:
                total_sold_sum += -row["Quantity"]
                proceeds_sum += row["Proceeds"]
                proceeds_conv_sum += row["Proceeds_converted"]
                comm_fee_sum += row["Comm/Fee"]
                basis_sum += row["Basis"]
                basis_conv_sum += row["Basis_converted"]
                comm_fee_conv_sum += row["Comm/Fee_converted"]
            elif row["Quantity"] > 0:
                fraction = row["fifo_allocated"] / row["Quantity"] if row["Quantity"] != 0 else 0
                proceeds_sum += row["Proceeds"] * fraction
                proceeds_conv_sum += row["Proceeds_converted"] * fraction
                comm_fee_sum += row["Comm/Fee"] * fraction
                basis_sum += row["Basis"] * fraction
                basis_conv_sum += row["Basis_converted"] * fraction
                comm_fee_conv_sum += row["Comm/Fee_converted"] * fraction

    summary = pd.DataFrame({
        "Stock": [group["Stock"].iloc[0]],
        "Total_Sold": [total_sold_sum],
        "Proceeds sum": [proceeds_sum],
        "Proceeds_converted sum": [proceeds_conv_sum],
        "Comm/Fee sum": [comm_fee_sum],
        "Basis sum": [basis_sum],
        "Basis_converted sum": [basis_conv_sum],
        "Comm/Fee_converted sum": [comm_fee_conv_sum]
    })
    return summary


def highlight_fifo(row):
    """
    Zwraca stylizację wiersza - zaznacza na żółto transakcje, które zostały wykorzystane.
    """
    return ['background-color: yellow' if row["fifo_used"] else '' for _ in row.index]


def check_negative_fifo(df_stock: pd.DataFrame) -> bool:
    """
    Sprawdza, czy dla danego stocku, idąc chronologicznie, bieżąca suma Quantity spada poniżej zera.
    Jeśli tak – zwraca True, co oznacza, że brakuje transakcji.
    """
    df_stock = df_stock.sort_values("Date/Time")
    running_total = 0.0
    for qty in df_stock["Quantity"]:
        running_total += qty
        if running_total < 0:
            return True
    return False


def process_all_trades() -> pd.DataFrame:
    """
    Przetwarza globalny DataFrame transakcji pełnym potokiem: filtrowanie, łączenie z kursami,
    konwersja walut oraz alokacja FIFO.
    """
    global all_trades_df
    if all_trades_df.empty:
        return pd.DataFrame()
    df = all_trades_df.copy()
    df = filter_and_convert_transactions(df)
    df_kursy = load_exchange_rates("kursy.csv")
    df = merge_exchange_rates(df, df_kursy)
    df = apply_currency_conversion(df)
    df = allocate_fifo(df)
    return df


# ----------------- Trasy Flask -----------------
@app.route("/", methods=["GET", "POST"])
def index():
    global all_trades_df, next_transaction_id

    if request.method == "POST":
        # Obsługa wgrywania plików HTML
        if "files" in request.files and any(file.filename for file in request.files.getlist("files")):
            files = request.files.getlist("files")
            df_list = []
            for file in files:
                if file.filename:
                    html_content = file.read().decode("utf-8")
                    try:
                        df = parse_html_transactions(html_content)
                    except ValueError as e:
                        return str(e), 400
                    df_list.append(df)
            if df_list:
                df_all = pd.concat(df_list, ignore_index=True)
                # Dodaj unikalny identyfikator do każdej transakcji
                df_all["id"] = range(next_transaction_id, next_transaction_id + len(df_all))
                next_transaction_id += len(df_all)
                if not all_trades_df.empty:
                    all_trades_df = pd.concat([all_trades_df, df_all], ignore_index=True)
                else:
                    all_trades_df = df_all.copy()
            return redirect(url_for("index"))
        # Obsługa dodawania transakcji z formularza wbudowanego w stronę wyników
        elif request.form.get("form_type") == "add_transaction":
            waluty = request.form.get("waluty")
            stock = request.form.get("Stock")
            date_time = request.form.get("DateTime")
            quantity = request.form.get("Quantity")
            proceeds = request.form.get("Proceeds")
            comm_fee = request.form.get("CommFee")
            basis = request.form.get("Basis")

            if not all([waluty, stock, date_time, quantity, proceeds, comm_fee, basis]):
                return "Wszystkie pola są wymagane", 400

            try:
                # Parsujemy datę z formatu ISO, generowanego przez datetime-local
                dt = datetime.fromisoformat(date_time)
            except ValueError:
                return "Nieprawidłowy format daty.", 400

            new_row = {
                "id": next_transaction_id,
                "waluty": waluty,
                "Stock": stock,
                "Date/Time": dt,
                "Quantity": quantity,
                "Proceeds": proceeds,
                "Comm/Fee": comm_fee,
                "Basis": basis
            }
            next_transaction_id += 1
            new_df = pd.DataFrame([new_row])
            if all_trades_df.empty:
                all_trades_df = new_df.copy()
            else:
                all_trades_df = pd.concat([all_trades_df, new_df], ignore_index=True)
            return redirect(url_for("index"))

    # Metoda GET – przetwarzamy transakcje i wyświetlamy wyniki
    processed_df = process_all_trades()
    stock_results = {}
    if not processed_df.empty:
        for stock, group in processed_df.groupby("Stock"):
            group_sorted = group.sort_values("Date/Time").copy()
            # Sprawdź, czy występuje ujemna suma transakcji dla danego stocku
            has_issue = check_negative_fifo(group_sorted)
            display_name = stock + (" !" if has_issue else "")
            group_sorted["Action"] = group_sorted["id"].apply(
                lambda x: f'<a href="{url_for("remove_transaction", transaction_id=x)}">Usuń</a>'
            )
            styled_group = group_sorted.style.apply(highlight_fifo, axis=1)
            transactions_html = styled_group.hide_index().render()
            summary = summarize_transactions(group_sorted)
            summary_html = summary.to_html(classes="table table-bordered", index=False, border=0)
            stock_results[stock] = {
                "display_name": display_name,
                "transactions": transactions_html,
                "summary": summary_html
            }
    return render_template("results.html", stock_results=stock_results)


@app.route("/remove-transaction/<int:transaction_id>")
def remove_transaction(transaction_id):
    global all_trades_df
    if not all_trades_df.empty:
        all_trades_df = all_trades_df[all_trades_df["id"] != transaction_id]
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
