from flask import Flask, render_template, request
import pandas as pd
import re
from bs4 import BeautifulSoup
from datetime import timedelta

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Odczyt pliku HTML przesłanego przez użytkownika
        if "file" not in request.files:
            return "Nie znaleziono pliku", 400
        file = request.files["file"]
        if file.filename == "":
            return "Nie wybrano pliku", 400
        html_content = file.read().decode("utf-8")

        # -----------------------------
        # Część 1: Parsowanie HTML i ekstrakcja transakcji
        # -----------------------------
        soup = BeautifulSoup(html_content, "html.parser")
        parent_container = soup.find(
            lambda tag: tag.name == "div" and tag.get("id") and re.search(r"^tblTransactions_.*Body$", tag.get("id")))
        if parent_container is None:
            return "Nie znaleziono kontenera z transakcjami", 400
        trades_table = parent_container.find("table")
        if trades_table is None:
            return "Nie znaleziono tabeli z transakcjami", 400

        data = []
        current_currency = None
        for tr in trades_table.find_all("tr"):
            cells = tr.find_all("td")
            # Jeśli wiersz ma jedną komórkę – ustawiamy walutę (podnagłówek)
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
        df_trades = pd.DataFrame(data)
        # Filtrowanie – usuwamy wiersze z pustym Quantity, te z "total" w Stock oraz te z pustym Basis
        df_trades = df_trades[~((df_trades["Quantity"].str.strip() == "") &
                                (df_trades["Stock"].str.lower().str.contains("total")))]
        df_trades = df_trades[df_trades["Basis"].str.strip() != ""]
        for col in ["Quantity", "Proceeds", "Comm/Fee", "Basis"]:
            df_trades[col] = pd.to_numeric(df_trades[col].str.replace(",", ""), errors="coerce")
        df_trades["Date/Time"] = pd.to_datetime(df_trades["Date/Time"], errors="coerce")

        # -----------------------------
        # Część 2: Ładowanie kursów i przeliczenia
        # -----------------------------
        df_kursy = pd.read_csv("kursy.csv", delimiter=",")
        df_kursy["data"] = pd.to_datetime(df_kursy["data"], format="%Y%m%d", errors="coerce")
        df_kursy = df_kursy.sort_values("data")
        # Odejmujemy 1 dobę od daty transakcji, aby wybrać kurs z dnia wcześniejszego
        df_trades["match_date"] = df_trades["Date/Time"] - timedelta(days=1)
        df_trades = df_trades.sort_values("match_date")
        df_trades = pd.merge_asof(df_trades, df_kursy, left_on="match_date", right_on="data", direction="backward")
        df_trades.rename(columns={"data": "Kurs_Date"}, inplace=True)
        df_trades.drop(columns=["match_date"], inplace=True)

        def wybierz_kurs(row):
            waluta = row["waluty"]
            if waluta == "USD":
                return row["1 USD"]
            elif waluta == "EUR":
                return row["1 EUR"]
            elif waluta == "GBP":
                return row["1 GBP"]
            elif waluta == "PLN":
                return 1.0
            else:
                return None

        df_trades["rate"] = df_trades.apply(wybierz_kurs, axis=1)
        df_trades["Basis_converted"] = df_trades["Basis"] * df_trades["rate"]
        df_trades["Comm/Fee_converted"] = df_trades["Comm/Fee"] * df_trades["rate"]
        df_trades["Proceeds_converted"] = df_trades["Proceeds"] * df_trades["rate"]
        df_trades = df_trades.drop(columns=["1 USD", "1 EUR", "1 GBP"], errors="ignore")

        desired_order = ["waluty", "Stock", "Date/Time", "Quantity", "Proceeds", "Proceeds_converted",
                         "Comm/Fee", "Basis", "Kurs_Date", "rate", "Basis_converted", "Comm/Fee_converted"]
        df_trades = df_trades[desired_order]
        df_trades.to_csv("trades.csv", index=False)  # opcjonalne zapisanie

        # -----------------------------
        # Część 3: Grupowanie po Stock, alokacja FIFO oraz podsumowanie
        # -----------------------------
        # Dodajemy kolumny pomocnicze do alokacji FIFO
        df_trades["fifo_allocated"] = 0.0  # dla kupna: ilość wykorzystana
        df_trades["fifo_used"] = False  # True, gdy transakcja (buy lub sell) została użyta

        # Sprzedaże (Quantity < 0) uznajemy za całkowicie użyte
        df_trades.loc[df_trades["Quantity"] < 0, "fifo_used"] = True

        stock_results = {}
        for stock, group in df_trades.groupby("Stock"):
            group = group.sort_values("Date/Time").copy()
            # Obliczamy całkowitą ilość sprzedaną (tylko sprzedaże, Quantity < 0)
            sells = group[group["Quantity"] < 0]
            total_sold = -sells["Quantity"].sum() if not sells.empty else 0.0

            # Alokacja kupna – FIFO: dla transakcji kupna, które nie są jeszcze wykorzystane
            buys = group[(group["Quantity"] > 0) & (group["fifo_used"] == False)].sort_values("Date/Time").copy()
            remaining = total_sold
            for idx, buy in buys.iterrows():
                if remaining <= 0:
                    break
                available = buy["Quantity"]
                if available <= remaining:
                    group.at[idx, "fifo_allocated"] = available
                    group.at[idx, "fifo_used"] = True
                    remaining -= available
                else:
                    group.at[idx, "fifo_allocated"] = remaining
                    group.at[idx, "fifo_used"] = True
                    remaining = 0

            # Podsumowanie FIFO – sumujemy dla wszystkich transakcji, które zostały użyte (fifo_used == True)
            total_sold_sum = 0.0  # suma sprzedanych (tylko ze sprzedaży)
            proceeds_sum = 0.0
            proceeds_conv_sum = 0.0
            comm_fee_sum = 0.0
            basis_sum = 0.0
            basis_conv_sum = 0.0
            comm_fee_conv_sum = 0.0

            for idx, row in group.iterrows():
                if row["fifo_used"]:
                    if row["Quantity"] < 0:
                        # Sprzedaż – używamy pełnych wartości
                        total_sold_sum += -row["Quantity"]
                        proceeds_sum += row["Proceeds"]
                        proceeds_conv_sum += row["Proceeds_converted"]
                        comm_fee_sum += row["Comm/Fee"]
                        # Dla sprzedaży dodajemy Basis, Basis_converted oraz Comm/Fee_converted (jeśli występują)
                        basis_sum += row["Basis"]
                        basis_conv_sum += row["Basis_converted"]
                        comm_fee_conv_sum += row["Comm/Fee_converted"]
                    elif row["Quantity"] > 0:
                        # Kupno – skalujemy wartości wg. udziału wykorzystanego (fifo_allocated / Quantity)
                        fraction = row["fifo_allocated"] / row["Quantity"] if row["Quantity"] != 0 else 0
                        # Kupna nie dodajemy do Total_Sold
                        proceeds_sum += row["Proceeds"] * fraction
                        proceeds_conv_sum += row["Proceeds_converted"] * fraction
                        comm_fee_sum += row["Comm/Fee"] * fraction
                        basis_sum += row["Basis"] * fraction
                        basis_conv_sum += row["Basis_converted"] * fraction
                        comm_fee_conv_sum += row["Comm/Fee_converted"] * fraction

            summary = pd.DataFrame({
                "Stock": [stock],
                "Total_Sold": [total_sold_sum],
                "Proceeds sum": [proceeds_sum],
                "Proceeds_converted sum": [proceeds_conv_sum],
                "Comm/Fee sum": [comm_fee_sum],
                "Basis sum": [basis_sum],
                "Basis_converted sum": [basis_conv_sum],
                "Comm/Fee_converted sum": [comm_fee_conv_sum]
            })

            # Oznaczamy na żółto transakcje użyte w FIFO
            def highlight_fifo(row):
                return ['background-color: yellow' if row["fifo_used"] else '' for _ in row.index]

            styled_group = group.style.apply(highlight_fifo, axis=1)
            transactions_html = styled_group.hide_index().render()
            summary_html = summary.to_html(classes="table table-bordered", index=False, border=0)
            stock_results[stock] = {
                "transactions": transactions_html,
                "summary": summary_html
            }

        return render_template("results.html", stock_results=stock_results)
    return '''
    <!doctype html>
    <html>
      <head>
        <title>Upload Transakcje HTML</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
      </head>
      <body>
        <div class="container">
          <h1>Wgraj plik HTML z transakcjami</h1>
          <form method="post" enctype="multipart/form-data">
            <div class="form-group">
              <input type="file" name="file" class="form-control-file">
            </div>
            <button type="submit" class="btn btn-primary">Prześlij</button>
          </form>
        </div>
      </body>
    </html>
    '''


if __name__ == "__main__":
    app.run(debug=True)
