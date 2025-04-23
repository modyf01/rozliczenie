from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import re
from bs4 import BeautifulSoup
from datetime import timedelta, datetime
import os

app = Flask(__name__)

# Globalny zbiór transakcji oraz licznik unikalnych identyfikatorów
all_trades_df = pd.DataFrame(
    columns=["id", "waluty", "Stock", "Date/Time", "Quantity", "Proceeds", "Comm/Fee", "Basis"]
)
next_transaction_id = 1
exchange_rates_file = "kursy.csv"  # Domyślna ścieżka do pliku z kursami


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
    df_trades["year_allocated"] = None  # Dodajemy nową kolumnę dla przypisania roku

    # Sprzedaże są zawsze w całości wykorzystane
    df_trades.loc[df_trades["Quantity"] < 0, "fifo_used"] = True
    df_trades.loc[df_trades["Quantity"] < 0, "year_allocated"] = df_trades.loc[df_trades["Quantity"] < 0, "Date/Time"].dt.year

    for stock, group in df_trades.groupby("Stock"):
        group = group.sort_values("Date/Time").copy()
        
        # Dla każdego roku, w którym wystąpiły sprzedaże, alokujemy kupna
        for year, year_group in group[group["Quantity"] < 0].groupby(group["Date/Time"].dt.year):
            sells_in_year = year_group[year_group["Quantity"] < 0]
            total_sold_in_year = -sells_in_year["Quantity"].sum() if not sells_in_year.empty else 0.0

            # Transakcje kupna, które nie są jeszcze w pełni wykorzystane
            buys = group[(group["Quantity"] > 0) & (group["Date/Time"] <= sells_in_year["Date/Time"].max())].sort_values("Date/Time").copy()
            remaining = total_sold_in_year
            
            for idx, buy in buys.iterrows():
                if remaining <= 0:
                    break
                    
                available = buy["Quantity"] - df_trades.at[idx, "fifo_allocated"]
                if available <= 0:
                    continue
                    
                if available <= remaining:
                    allocated_to_year = available
                    remaining -= available
                else:
                    allocated_to_year = remaining
                    remaining = 0
                
                # Dodajemy alokację dla danego roku
                if pd.isna(df_trades.at[idx, "year_allocated"]):
                    df_trades.at[idx, "year_allocated"] = {}
                elif isinstance(df_trades.at[idx, "year_allocated"], (int, float)):
                    # Konwersja z pojedynczego roku na słownik
                    df_trades.at[idx, "year_allocated"] = {int(df_trades.at[idx, "year_allocated"]): df_trades.at[idx, "fifo_allocated"]}
                
                # Aktualizujemy alokację dla danego roku
                year_alloc = df_trades.at[idx, "year_allocated"]
                if isinstance(year_alloc, dict):
                    if year in year_alloc:
                        year_alloc[year] += allocated_to_year
                    else:
                        year_alloc[year] = allocated_to_year
                    df_trades.at[idx, "year_allocated"] = year_alloc
                else:
                    df_trades.at[idx, "year_allocated"] = {year: allocated_to_year}
                
                # Aktualizujemy łączną alokację
                df_trades.at[idx, "fifo_allocated"] += allocated_to_year
                if df_trades.at[idx, "fifo_allocated"] >= buy["Quantity"]:
                    df_trades.at[idx, "fifo_used"] = True
    
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


def summarize_transactions_by_year(group: pd.DataFrame) -> pd.DataFrame:
    """
    Generuje podsumowanie transakcji dla danego symbolu akcji z podziałem na lata.
    """
    # Zbieramy wszystkie lata, w których wystąpiły transakcje sprzedaży
    years = []
    for idx, row in group.iterrows():
        if row["Quantity"] < 0:  # Transakcja sprzedaży
            year = row["Date/Time"].year
            if year not in years:
                years.append(year)
        elif row["Quantity"] > 0 and isinstance(row["year_allocated"], dict):  # Transakcja kupna z alokacją roczną
            for year in row["year_allocated"].keys():
                if year not in years:
                    years.append(year)
    
    years.sort()
    
    # Przygotowujemy DataFrame na podsumowanie roczne
    year_summary_data = []
    
    for year in years:
        total_sold_year = 0.0
        proceeds_year = 0.0
        proceeds_conv_year = 0.0
        comm_fee_year = 0.0
        basis_year = 0.0
        basis_conv_year = 0.0
        comm_fee_conv_year = 0.0
        
        for idx, row in group.iterrows():
            if row["Quantity"] < 0 and row["Date/Time"].year == year:  # Transakcja sprzedaży w danym roku
                total_sold_year += -row["Quantity"]
                proceeds_year += row["Proceeds"]
                proceeds_conv_year += row["Proceeds_converted"]
                comm_fee_year += row["Comm/Fee"]
            elif row["Quantity"] > 0 and isinstance(row["year_allocated"], dict) and year in row["year_allocated"]:
                # Transakcja kupna z alokacją do danego roku
                fraction = row["year_allocated"][year] / row["Quantity"] if row["Quantity"] != 0 else 0
                basis_year += row["Basis"] * fraction
                basis_conv_year += row["Basis_converted"] * fraction
                comm_fee_conv_year += row["Comm/Fee_converted"] * fraction
        
        year_summary_data.append({
            "Year": year,
            "Stock": group["Stock"].iloc[0],
            "Total_Sold": total_sold_year,
            "Proceeds sum": proceeds_year,
            "Proceeds_converted sum": proceeds_conv_year,
            "Comm/Fee sum": comm_fee_year,
            "Basis sum": basis_year,
            "Basis_converted sum": basis_conv_year,
            "Comm/Fee_converted sum": comm_fee_conv_year
        })
    
    if year_summary_data:
        return pd.DataFrame(year_summary_data)
    else:
        return pd.DataFrame()


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
    global all_trades_df, exchange_rates_file
    if all_trades_df.empty:
        return pd.DataFrame()
    df = all_trades_df.copy()
    df = filter_and_convert_transactions(df)
    df_kursy = load_exchange_rates(exchange_rates_file)
    df = merge_exchange_rates(df, df_kursy)
    df = apply_currency_conversion(df)
    df = allocate_fifo(df)
    return df


# ----------------- Trasy Flask -----------------
@app.route("/", methods=["GET", "POST"])
def index():
    global all_trades_df, next_transaction_id, exchange_rates_file

    if request.method == "POST":
        # Obsługa wgrywania pliku CSV z kursami walut
        if "exchange_rates_file" in request.files and request.files["exchange_rates_file"].filename:
            file = request.files["exchange_rates_file"]
            if file.filename.endswith('.csv'):
                # Zapisz plik tymczasowo
                temp_path = "uploaded_" + file.filename
                file.save(temp_path)
                
                try:
                    # Sprawdź czy plik ma prawidłowy format
                    test_df = pd.read_csv(temp_path, delimiter=",")
                    required_columns = ["data", "1 USD", "1 EUR", "1 GBP"]
                    if all(col in test_df.columns for col in required_columns):
                        exchange_rates_file = temp_path
                    else:
                        os.remove(temp_path)
                        return "Plik CSV musi zawierać kolumny: data, 1 USD, 1 EUR, 1 GBP", 400
                except Exception as e:
                    os.remove(temp_path)
                    return f"Błąd podczas przetwarzania pliku CSV: {str(e)}", 400
                
            else:
                return "Plik musi mieć rozszerzenie .csv", 400
            
            return redirect(url_for("index"))
        
        # Obsługa wgrywania plików HTML
        elif "files" in request.files and any(file.filename for file in request.files.getlist("files")):
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
    yearly_summaries = {}  # Dodajemy słownik na podsumowania roczne
    
    if not processed_df.empty:
        # Dodajemy obliczenie sum dla wszystkich akcji
        all_summary = pd.DataFrame({
            "Stock": ["All"],
            "Total_Sold": [0.0],
            "Proceeds sum": [0.0],
            "Proceeds_converted sum": [0.0],
            "Comm/Fee sum": [0.0],
            "Basis sum": [0.0],
            "Basis_converted sum": [0.0],
            "Comm/Fee_converted sum": [0.0]
        })
        
        # Przygotowanie słownika na roczne podsumowanie dla zakładki "All"
        all_years_summary = {}
        
        for stock, group in processed_df.groupby("Stock"):
            group_sorted = group.sort_values("Date/Time").copy()
            # Sprawdź, czy występuje ujemna suma transakcji dla danego stocku
            has_issue = check_negative_fifo(group_sorted)
            display_name = stock + (" !" if has_issue else "")
            group_sorted["Action"] = group_sorted["id"].apply(
                lambda x: f'<a href="{url_for("remove_transaction", transaction_id=x)}">Usuń</a>'
            )
            
            # Zmiana formatowania liczb w tabeli transakcji
            styled_group = group_sorted.style.apply(highlight_fifo, axis=1).format({
                col: lambda x: f"{x:.6f}" if isinstance(x, (float, int)) else x 
                for col in group_sorted.select_dtypes(include=['float64', 'int64']).columns
            })
            
            transactions_html = styled_group.hide_index().render()
            summary = summarize_transactions(group_sorted)
            
            # Zmiana formatowania liczb w tabeli podsumowania
            summary_html = summary.to_html(
                classes="table table-bordered", 
                index=False, 
                border=0,
                float_format=lambda x: f"{x:.6f}"
            )
            
            # Dodajemy podsumowanie roczne
            yearly_summary = summarize_transactions_by_year(group_sorted)
            if not yearly_summary.empty:
                yearly_summary_html = yearly_summary.to_html(
                    classes="table table-bordered", 
                    index=False, 
                    border=0,
                    float_format=lambda x: f"{x:.6f}" if isinstance(x, (float, int)) else x
                )
                yearly_summaries[stock] = yearly_summary_html
                
                # Dodajemy dane do podsumowania rocznego dla zakładki "All"
                for _, row in yearly_summary.iterrows():
                    year = row["Year"]
                    if year not in all_years_summary:
                        all_years_summary[year] = {
                            "Total_Sold": 0.0,
                            "Proceeds sum": 0.0,
                            "Proceeds_converted sum": 0.0,
                            "Comm/Fee sum": 0.0,
                            "Basis sum": 0.0,
                            "Basis_converted sum": 0.0,
                            "Comm/Fee_converted sum": 0.0
                        }
                    all_years_summary[year]["Total_Sold"] += row["Total_Sold"]
                    all_years_summary[year]["Proceeds sum"] += row["Proceeds sum"]
                    all_years_summary[year]["Proceeds_converted sum"] += row["Proceeds_converted sum"]
                    all_years_summary[year]["Comm/Fee sum"] += row["Comm/Fee sum"]
                    all_years_summary[year]["Basis sum"] += row["Basis sum"]
                    all_years_summary[year]["Basis_converted sum"] += row["Basis_converted sum"]
                    all_years_summary[year]["Comm/Fee_converted sum"] += row["Comm/Fee_converted sum"]
            
            stock_results[stock] = {
                "display_name": display_name,
                "transactions": transactions_html,
                "summary": summary_html
            }
            
            # Aktualizujemy sumy dla zakładki "All"
            if not summary.empty:
                all_summary["Proceeds_converted sum"] += summary["Proceeds_converted sum"].values[0]
                all_summary["Basis_converted sum"] += summary["Basis_converted sum"].values[0]
                all_summary["Comm/Fee_converted sum"] += summary["Comm/Fee_converted sum"].values[0]
        
        # Tworzymy DataFrame z podsumowaniem rocznym dla zakładki "All"
        if all_years_summary:
            all_yearly_data = []
            for year, data in sorted(all_years_summary.items()):
                row_data = {"Year": year}
                row_data.update(data)
                all_yearly_data.append(row_data)
            
            all_yearly_df = pd.DataFrame(all_yearly_data)
            all_yearly_html = all_yearly_df.to_html(
                classes="table table-bordered", 
                index=False, 
                border=0,
                float_format=lambda x: f"{x:.6f}" if isinstance(x, (float, int)) else x
            )
            yearly_summaries["all"] = all_yearly_html
        
        # Dodajemy zakładkę "All" do wyników
        # Tworzymy nowy DataFrame tylko z potrzebnymi kolumnami
        all_summary_display = pd.DataFrame({
            "Proceeds_converted sum": [all_summary["Proceeds_converted sum"].values[0]],
            "Basis_converted sum": [all_summary["Basis_converted sum"].values[0]],
            "Comm/Fee_converted sum": [all_summary["Comm/Fee_converted sum"].values[0]]
        })
        
        # Zmiana formatowania liczb w tabeli podsumowania "All"
        all_summary_html = all_summary_display.to_html(
            classes="table table-bordered", 
            index=False, 
            border=0,
            float_format=lambda x: f"{x:.6f}"
        )
        
        stock_results["all"] = {
            "display_name": "All",
            "transactions": "",  # Puste, bo nie pokazujemy transakcji w tej zakładce
            "summary": all_summary_html
        }
    
    # Dodaj informację o obecnie używanym pliku z kursami
    current_rates_file = os.path.basename(exchange_rates_file)
    
    return render_template("results.html", stock_results=stock_results, 
                           yearly_summaries=yearly_summaries, current_rates_file=current_rates_file)


@app.route("/remove-transaction/<int:transaction_id>")
def remove_transaction(transaction_id):
    global all_trades_df
    if not all_trades_df.empty:
        all_trades_df = all_trades_df[all_trades_df["id"] != transaction_id]
    return redirect(url_for("index"))


@app.route("/refresh")
def refresh():
    """
    Odświeża aplikację - czyści wszystkie załadowane transakcje.
    """
    global all_trades_df, next_transaction_id, exchange_rates_file
    all_trades_df = pd.DataFrame(
        columns=["id", "waluty", "Stock", "Date/Time", "Quantity", "Proceeds", "Comm/Fee", "Basis"]
    )
    next_transaction_id = 1
    # Resetowanie do domyślnego pliku z kursami
    if exchange_rates_file != "kursy.csv" and os.path.exists(exchange_rates_file):
        if exchange_rates_file.startswith("uploaded_"):
            os.remove(exchange_rates_file)
    exchange_rates_file = "kursy.csv"
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
