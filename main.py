from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import re
from bs4 import BeautifulSoup
from datetime import timedelta, datetime
import os
import io

app = Flask(__name__)

# Globalny zbiór transakcji oraz licznik unikalnych identyfikatorów
all_trades_df = pd.DataFrame(
    columns=["id", "waluty", "Stock", "Date/Time", "Quantity", "Proceeds", "Comm/Fee", "Basis", "shares_in_possession"]
)
next_transaction_id = 1
exchange_rates_file = "kursy.csv"  # Domyślna ścieżka do pliku z kursami


# ----------------- Funkcje pomocnicze -----------------
def parse_html_transactions(html_content: str) -> pd.DataFrame:
    """
    Parsuje HTML i zwraca DataFrame z danymi transakcji.
    Obsługuje różne formaty tabel poprzez wykrywanie nagłówków.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    parent_container = soup.find(
        lambda tag: tag.name == "div" and tag.get("id") and re.search(r"^tblTransactions_.*Body$", tag.get("id"))
    )
    
    # Jeśli nie znaleziono standardowego kontenera, szukamy alternatywnych struktur
    if parent_container is None:
        # Szukaj tabeli w całym dokumencie
        trades_table = soup.find("table", {"class": "table-bordered"})
        if trades_table is None:
            raise ValueError("Nie znaleziono tabeli z transakcjami")
    else:
        # Znajdź tabelę z transakcjami w standardowym kontenerze
        trades_table = parent_container.find("table")
        if trades_table is None:
            raise ValueError("Nie znaleziono tabeli z transakcjami")

    # Znajdź nagłówki kolumn, aby określić ich indeksy
    column_indices = {}
    headers = trades_table.find_all("th")
    
    if headers:
        for i, header in enumerate(headers):
            header_text = header.get_text(strip=True).lower()
            if "symbol" in header_text:
                column_indices["symbol"] = i
            elif "date/time" in header_text:
                column_indices["date_time"] = i
            elif "quantity" in header_text:
                column_indices["quantity"] = i
            elif "proceeds" in header_text:
                column_indices["proceeds"] = i
            elif "comm/fee" in header_text:
                column_indices["comm_fee"] = i
            elif "basis" in header_text:
                column_indices["basis"] = i
            elif "account" in header_text:
                column_indices["account"] = i
            if all(key in column_indices for key in ["symbol", "date_time", "quantity", "proceeds", "comm_fee", "basis"]):
                break
    
    # Jeśli nie znaleziono nagłówków, używamy domyślnego układu
    if not column_indices:
        # Domyślny układ (z pierwszego formatu)
        column_indices = {
            "symbol": 0,
            "date_time": 1,
            "quantity": 2,
            "proceeds": 5,
            "comm_fee": 6,
            "basis": 7
        }

    data = []
    current_currency = None
    
    # Sprawdź, czy mamy układ z numerem konta
    has_account_column = "account" in column_indices
    
    for tr in trades_table.find_all("tr"):
        cells = tr.find_all("td")
        
        # Wiersze z jedną komórką traktujemy jako nagłówek waluty
        if len(cells) == 1:
            text = cells[0].get_text(strip=True)
            if text in ["EUR", "GBP", "USD", "PLN"]:
                current_currency = text
            continue
        
        # Sprawdź, czy wiersz zawiera dane transakcji (musi mieć wystarczająco kolumn)
        if len(cells) >= 8:
            try:
                # Obsługa przypadku, gdy pierwsza kolumna to "Account"
                if has_account_column:
                    # Jeśli mamy wykrytą kolumnę account, używamy ją jako wskaźnik
                    account_idx = column_indices.get("account", 0)
                    symbol_idx = account_idx + 1  # Symbol jest następną kolumną po Account
                    date_time_idx = account_idx + 2
                    quantity_idx = account_idx + 3
                    # Pozostałe indeksy dostosowujemy według układu
                    proceeds_idx = account_idx + 6  # Zakładamy, że Proceeds jest 6 kolumn po Account
                    comm_fee_idx = account_idx + 7
                    basis_idx = account_idx + 8
                else:
                    # Pobierz dane z komórek zgodnie z wykrytymi indeksami kolumn
                    symbol_idx = column_indices.get("symbol", 0)
                    date_time_idx = column_indices.get("date_time", 1)
                    quantity_idx = column_indices.get("quantity", 2)
                    proceeds_idx = column_indices.get("proceeds", 5)
                    comm_fee_idx = column_indices.get("comm_fee", 6)
                    basis_idx = column_indices.get("basis", 7)
                
                # Zabezpieczenie przed wyjściem poza zakres
                symbol_idx = min(symbol_idx, len(cells) - 1)
                stock = cells[symbol_idx].get_text(strip=True)
                
                # Mapuj FB na META, ponieważ to ten sam stock
                if stock == "FB":
                    stock = "META"
                
                # Zabezpieczenie przed wyjściem poza zakres
                date_time_idx = min(date_time_idx, len(cells) - 1)
                quantity_idx = min(quantity_idx, len(cells) - 1)
                proceeds_idx = min(proceeds_idx, len(cells) - 1)
                comm_fee_idx = min(comm_fee_idx, len(cells) - 1)
                basis_idx = min(basis_idx, len(cells) - 1)
                
                date_time = cells[date_time_idx].get_text(strip=True)
                quantity = cells[quantity_idx].get_text(strip=True)
                proceeds = cells[proceeds_idx].get_text(strip=True)
                comm_fee = cells[comm_fee_idx].get_text(strip=True)
                basis = cells[basis_idx].get_text(strip=True)

                if not stock.startswith("Total") and not cells[0].get_text(strip=True).startswith("Total"):
                    data.append({
                        "waluty": current_currency,
                        "Stock": stock,
                        "Date/Time": date_time,
                        "Quantity": quantity,
                        "Proceeds": proceeds,
                        "Comm/Fee": comm_fee,
                        "Basis": basis
                    })
            except Exception as e:
                # Logowanie błędu do konsoli - możesz zakomentować lub usunąć w produkcji
                print(f"Błąd przetwarzania wiersza: {e}")
                continue
    
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
                     "Comm/Fee", "Basis", "Kurs_Date", "rate", "Basis_converted", "Comm/Fee_converted", 
                     "shares_in_possession"]
    return df_trades[desired_order]


def allocate_fifo(df_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Alokacja FIFO – przypisuje transakcjom kupna i sprzedaży wykorzystanie 
    zgodnie z kolejnością transakcji. Obsługuje zarówno pozycje długie 
    jak i krótką sprzedaż (short selling).
    """
    df_trades["fifo_allocated"] = 0.0
    df_trades["fifo_used"] = False
    df_trades["year_allocated"] = None
    df_trades["shares_in_possession"] = 0.0  # Nowa kolumna do śledzenia posiadanych akcji

    for stock, group in df_trades.groupby("Stock"):
        # Sortujemy transakcje chronologicznie
        group = group.sort_values("Date/Time").copy()
        
        # Tworzymy oddzielne kolejki dla transakcji kupna i sprzedaży
        buy_queue = []  # Format: (index, ilość pozostała do alokacji, oryginalna ilość)
        sell_queue = []  # Format: (index, ilość pozostała do alokacji, oryginalna ilość)
        
        # Zmienna do śledzenia aktualnie posiadanych akcji dla danego stocku
        current_possession = 0.0
        
        # Przetwarzamy transakcje chronologicznie
        for idx, row in group.iterrows():
            quantity = row["Quantity"]
            transaction_year = row["Date/Time"].year
            
            # Aktualizujemy stan posiadania akcji
            current_possession += quantity
            # Zapisujemy aktualny stan posiadania dla tej transakcji
            df_trades.at[idx, "shares_in_possession"] = current_possession
            
            if quantity > 0:  # Transakcja kupna
                # Najpierw próbujemy pokryć istniejące pozycje krótkie
                remaining_buy = quantity
                
                while sell_queue and remaining_buy > 0:
                    sell_idx, sell_remaining, sell_original = sell_queue[0]
                    
                    # Określamy ile można alokować
                    to_allocate = min(remaining_buy, sell_remaining)
                    
                    # Aktualizujemy stan transakcji sprzedaży
                    sell_queue[0] = (sell_idx, sell_remaining - to_allocate, sell_original)
                    
                    # Aktualizujemy flagi dla transakcji sprzedaży
                    df_trades.at[sell_idx, "fifo_allocated"] += to_allocate
                    if sell_remaining == to_allocate:  # Całkowicie pokryta
                        df_trades.at[sell_idx, "fifo_used"] = True
                        sell_queue.pop(0)  # Usuwamy z kolejki
                    
                    # Aktualizujemy flagi dla transakcji kupna
                    df_trades.at[idx, "fifo_allocated"] += to_allocate
                    
                    # Przypisanie roku dla celów podatkowych
                    # Dla transakcji sprzedaży (short selling)
                    if pd.isna(df_trades.at[sell_idx, "year_allocated"]):
                        df_trades.at[sell_idx, "year_allocated"] = {transaction_year: to_allocate}
                    elif isinstance(df_trades.at[sell_idx, "year_allocated"], dict):
                        if transaction_year in df_trades.at[sell_idx, "year_allocated"]:
                            df_trades.at[sell_idx, "year_allocated"][transaction_year] += to_allocate
                        else:
                            df_trades.at[sell_idx, "year_allocated"][transaction_year] = to_allocate
                    else:
                        # Konwersja z pojedynczego roku na słownik
                        old_year = df_trades.at[sell_idx, "year_allocated"]
                        old_allocated = sell_original - sell_remaining + to_allocate
                        df_trades.at[sell_idx, "year_allocated"] = {
                            old_year: old_allocated - to_allocate,
                            transaction_year: to_allocate
                        }
                    
                    # Dla transakcji kupna
                    if pd.isna(df_trades.at[idx, "year_allocated"]):
                        df_trades.at[idx, "year_allocated"] = {transaction_year: to_allocate}
                    elif isinstance(df_trades.at[idx, "year_allocated"], dict):
                        if transaction_year in df_trades.at[idx, "year_allocated"]:
                            df_trades.at[idx, "year_allocated"][transaction_year] += to_allocate
                        else:
                            df_trades.at[idx, "year_allocated"][transaction_year] = to_allocate
                    else:
                        # Konwersja z pojedynczego roku na słownik
                        df_trades.at[idx, "year_allocated"] = {transaction_year: to_allocate}
                    
                    remaining_buy -= to_allocate
                
                # Jeśli zostało coś niezaalokowanego, dodajemy do kolejki kupna
                if remaining_buy > 0:
                    buy_queue.append((idx, remaining_buy, quantity))
                
                # Jeśli całkowicie zaalokowane
                if remaining_buy == 0:
                    df_trades.at[idx, "fifo_used"] = True
                
            else:  # Transakcja sprzedaży (quantity < 0)
                # Absolutna wartość ilości sprzedaży
                abs_quantity = -quantity
                remaining_sell = abs_quantity
                
                # Najpierw próbujemy pokryć istniejące pozycje długie
                while buy_queue and remaining_sell > 0:
                    buy_idx, buy_remaining, buy_original = buy_queue[0]
                    
                    # Określamy ile można alokować
                    to_allocate = min(remaining_sell, buy_remaining)
                    
                    # Aktualizujemy stan transakcji kupna
                    buy_queue[0] = (buy_idx, buy_remaining - to_allocate, buy_original)
                    
                    # Aktualizujemy flagi dla transakcji kupna
                    df_trades.at[buy_idx, "fifo_allocated"] += to_allocate
                    if buy_remaining == to_allocate:  # Całkowicie pokryta
                        df_trades.at[buy_idx, "fifo_used"] = True
                        buy_queue.pop(0)  # Usuwamy z kolejki
                    
                    # Aktualizujemy flagi dla transakcji sprzedaży
                    df_trades.at[idx, "fifo_allocated"] += to_allocate
                    
                    # Przypisanie roku dla celów podatkowych
                    # Dla transakcji sprzedaży
                    if pd.isna(df_trades.at[idx, "year_allocated"]):
                        df_trades.at[idx, "year_allocated"] = {transaction_year: to_allocate}
                    elif isinstance(df_trades.at[idx, "year_allocated"], dict):
                        if transaction_year in df_trades.at[idx, "year_allocated"]:
                            df_trades.at[idx, "year_allocated"][transaction_year] += to_allocate
                        else:
                            df_trades.at[idx, "year_allocated"][transaction_year] = to_allocate
                    else:
                        # Konwersja z pojedynczego roku na słownik
                        df_trades.at[idx, "year_allocated"] = {transaction_year: to_allocate}
                    
                    # Dla transakcji kupna
                    if pd.isna(df_trades.at[buy_idx, "year_allocated"]):
                        df_trades.at[buy_idx, "year_allocated"] = {transaction_year: to_allocate}
                    elif isinstance(df_trades.at[buy_idx, "year_allocated"], dict):
                        if transaction_year in df_trades.at[buy_idx, "year_allocated"]:
                            df_trades.at[buy_idx, "year_allocated"][transaction_year] += to_allocate
                        else:
                            df_trades.at[buy_idx, "year_allocated"][transaction_year] = to_allocate
                    else:
                        # Konwersja z pojedynczego roku na słownik
                        old_year = df_trades.at[buy_idx, "year_allocated"]
                        old_allocated = buy_original - buy_remaining + to_allocate
                        df_trades.at[buy_idx, "year_allocated"] = {
                            old_year: old_allocated - to_allocate,
                            transaction_year: to_allocate
                        }
                    
                    remaining_sell -= to_allocate
                
                # Jeśli zostało coś niezaalokowanego, dodajemy do kolejki sprzedaży (short selling)
                if remaining_sell > 0:
                    sell_queue.append((idx, remaining_sell, abs_quantity))
                
                # Jeśli całkowicie zaalokowane
                if remaining_sell == 0:
                    df_trades.at[idx, "fifo_used"] = True
    
    return df_trades


def summarize_transactions(group: pd.DataFrame) -> pd.DataFrame:
    """
    Generuje podsumowanie transakcji dla danego symbolu akcji.
    Uwzględnia proporcjonalne wykorzystanie transakcji w zależności od 
    wartości fifo_allocated.
    """
    total_sold_sum = 0.0
    proceeds_sum = 0.0
    proceeds_conv_sum = 0.0
    comm_fee_sum = 0.0
    basis_sum = 0.0
    basis_conv_sum = 0.0
    comm_fee_conv_sum = 0.0
    
    # Nowe sumy dla transakcji sprzedaży (quantity < 0)
    proceeds_sum_sell = 0.0
    proceeds_conv_sum_sell = 0.0
    comm_fee_sum_sell = 0.0
    basis_sum_sell = 0.0
    basis_conv_sum_sell = 0.0
    comm_fee_conv_sum_sell = 0.0

    for idx, row in group.iterrows():
        # Pomijamy transakcje bez alokacji
        if row["fifo_allocated"] <= 0:
            continue
            
        # Obliczamy proporcję wykorzystania transakcji
        proportion = row["fifo_allocated"] / abs(row["Quantity"]) if row["Quantity"] != 0 else 0
        
        # Dodajemy wartości do sum, niezależnie od typu transakcji
        if row["Quantity"] < 0:  # Transakcja sprzedaży
            total_sold_sum += row["fifo_allocated"]  # Używamy dokładnej wartości alokowanej
            
            # Dodajemy wartości do sum dla transakcji sprzedaży
            proceeds_sum_sell += row["Proceeds"] * proportion
            proceeds_conv_sum_sell += row["Proceeds_converted"] * proportion
            comm_fee_sum_sell += row["Comm/Fee"] * proportion
            basis_sum_sell += row["Basis"] * proportion
            basis_conv_sum_sell += row["Basis_converted"] * proportion
            comm_fee_conv_sum_sell += row["Comm/Fee_converted"] * proportion
        
        # Wszystkie transakcje (zarówno kupna jak i sprzedaży) dodają wartości do wszystkich sum
        proceeds_sum += row["Proceeds"] * proportion
        proceeds_conv_sum += row["Proceeds_converted"] * proportion
        comm_fee_sum += row["Comm/Fee"] * proportion
        basis_sum += row["Basis"] * proportion
        basis_conv_sum += row["Basis_converted"] * proportion
        comm_fee_conv_sum += row["Comm/Fee_converted"] * proportion

    summary = pd.DataFrame({
        "Stock": [group["Stock"].iloc[0]],
        "Total_Sold": [total_sold_sum],
        "Proceeds sum": [proceeds_sum],
        "Proceeds_converted sum": [proceeds_conv_sum],
        "Comm/Fee sum": [comm_fee_sum],
        "Basis sum": [basis_sum],
        "Basis_converted sum": [basis_conv_sum],
        "Comm/Fee_converted sum": [comm_fee_conv_sum],
        # Dodajemy nowe kolumny dla transakcji sprzedaży
        "Proceeds sum (quantity < 0)": [proceeds_sum_sell],
        "Proceeds_converted sum (quantity < 0)": [proceeds_conv_sum_sell],
        "Comm/Fee sum (quantity < 0)": [comm_fee_sum_sell],
        "Basis sum (quantity < 0)": [basis_sum_sell],
        "Basis_converted sum (quantity < 0)": [basis_conv_sum_sell],
        "Comm/Fee_converted sum (quantity < 0)": [comm_fee_conv_sum_sell]
    })
    return summary


def summarize_transactions_by_year(group: pd.DataFrame) -> pd.DataFrame:
    """
    Generuje podsumowanie transakcji dla danego symbolu akcji z podziałem na lata.
    """
    # Zbieramy wszystkie lata, w których wystąpiły transakcje
    years = set()
    
    for idx, row in group.iterrows():
        if isinstance(row["year_allocated"], dict):
            # Dla transakcji z przypisanym słownikiem lat
            for year in row["year_allocated"].keys():
                years.add(year)
        elif pd.notna(row["year_allocated"]):
            # Dla transakcji z pojedynczym przypisanym rokiem
            years.add(int(row["year_allocated"]))
    
    years = sorted(years)
    
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
        
        # Nowe sumy dla transakcji sprzedaży (quantity < 0)
        proceeds_year_sell = 0.0
        proceeds_conv_year_sell = 0.0
        comm_fee_year_sell = 0.0
        basis_year_sell = 0.0
        basis_conv_year_sell = 0.0
        comm_fee_conv_year_sell = 0.0
        
        for idx, row in group.iterrows():
            # Pomijamy transakcje bez alokacji dla danego roku
            year_allocation = row["year_allocated"]
            if not isinstance(year_allocation, dict) or year not in year_allocation:
                if not (pd.notna(year_allocation) and year_allocation == year):
                    continue
            
            # Obliczamy proporcję alokowaną do danego roku
            if isinstance(year_allocation, dict) and year in year_allocation:
                allocated_to_year = year_allocation[year]
                total_fraction = allocated_to_year / abs(row["Quantity"]) if row["Quantity"] != 0 else 0
                
                # Zliczamy sprzedane akcje
                if row["Quantity"] < 0:  # Transakcja sprzedaży
                    total_sold_year += allocated_to_year
                    
                    # Dodajemy wartości do sum dla transakcji sprzedaży
                    proceeds_year_sell += row["Proceeds"] * total_fraction
                    proceeds_conv_year_sell += row["Proceeds_converted"] * total_fraction
                    comm_fee_year_sell += row["Comm/Fee"] * total_fraction
                    basis_year_sell += row["Basis"] * total_fraction
                    basis_conv_year_sell += row["Basis_converted"] * total_fraction
                    comm_fee_conv_year_sell += row["Comm/Fee_converted"] * total_fraction
                
                # Wszystkie transakcje dodają wartości do wszystkich sum
                proceeds_year += row["Proceeds"] * total_fraction
                proceeds_conv_year += row["Proceeds_converted"] * total_fraction
                comm_fee_year += row["Comm/Fee"] * total_fraction
                basis_year += row["Basis"] * total_fraction
                basis_conv_year += row["Basis_converted"] * total_fraction
                comm_fee_conv_year += row["Comm/Fee_converted"] * total_fraction
            elif pd.notna(year_allocation) and year_allocation == year:
                # Dla zgodności wstecz - obsługa prostego przypisania roku
                if row["Quantity"] < 0:  # Transakcja sprzedaży
                    total_sold_year += -row["Quantity"]
                    
                    # Dodajemy wartości do sum dla transakcji sprzedaży
                    proceeds_year_sell += row["Proceeds"]
                    proceeds_conv_year_sell += row["Proceeds_converted"]
                    comm_fee_year_sell += row["Comm/Fee"]
                    basis_year_sell += row["Basis"]
                    basis_conv_year_sell += row["Basis_converted"]
                    comm_fee_conv_year_sell += row["Comm/Fee_converted"]
                
                # Dodajemy wszystkie wartości (niezależnie od typu transakcji)
                proceeds_year += row["Proceeds"]
                proceeds_conv_year += row["Proceeds_converted"]
                comm_fee_year += row["Comm/Fee"]
                basis_year += row["Basis"]
                basis_conv_year += row["Basis_converted"]
                comm_fee_conv_year += row["Comm/Fee_converted"]
        
        year_summary_data.append({
            "Year": year,
            "Stock": group["Stock"].iloc[0],
            "Total_Sold": total_sold_year,
            "Proceeds sum": proceeds_year,
            "Proceeds_converted sum": proceeds_conv_year,
            "Comm/Fee sum": comm_fee_year,
            "Basis sum": basis_year,
            "Basis_converted sum": basis_conv_year,
            "Comm/Fee_converted sum": comm_fee_conv_year,
            # Dodajemy nowe kolumny dla transakcji sprzedaży
            "Proceeds sum (quantity < 0)": proceeds_year_sell,
            "Proceeds_converted sum (quantity < 0)": proceeds_conv_year_sell,
            "Comm/Fee sum (quantity < 0)": comm_fee_year_sell,
            "Basis sum (quantity < 0)": basis_year_sell,
            "Basis_converted sum (quantity < 0)": basis_conv_year_sell,
            "Comm/Fee_converted sum (quantity < 0)": comm_fee_conv_year_sell
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


def standardize_stock_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standaryzuje symbole akcji - zamienia FB na META
    """
    df.loc[df["Stock"] == "FB", "Stock"] = "META"
    return df


def process_all_trades() -> pd.DataFrame:
    """
    Przetwarza globalny DataFrame transakcji pełnym potokiem: filtrowanie, łączenie z kursami,
    konwersja walut oraz alokacja FIFO.
    """
    global all_trades_df, exchange_rates_file
    if all_trades_df.empty:
        return pd.DataFrame()
    df = all_trades_df.copy()
    
    # Upewniamy się, że kolumna shares_in_possession istnieje
    if "shares_in_possession" not in df.columns:
        df["shares_in_possession"] = 0.0
    
    # Standaryzujemy symbole akcji
    df = standardize_stock_symbols(df)
        
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
            
            # Standardize stock symbol
            if stock == "FB":
                stock = "META"
            
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
                "Basis": basis,
                "shares_in_possession": 0.0  # Inicjalizujemy nową kolumnę
            }
            next_transaction_id += 1
            new_df = pd.DataFrame([new_row])
            if all_trades_df.empty:
                all_trades_df = new_df.copy()
            else:
                # Upewniamy się, że kolumna shares_in_possession istnieje w all_trades_df
                if "shares_in_possession" not in all_trades_df.columns:
                    all_trades_df["shares_in_possession"] = 0.0
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
            "Comm/Fee_converted sum": [0.0],
            # Dodajemy nowe kolumny dla transakcji sprzedaży
            "Proceeds sum (quantity < 0)": [0.0],
            "Proceeds_converted sum (quantity < 0)": [0.0],
            "Comm/Fee sum (quantity < 0)": [0.0],
            "Basis sum (quantity < 0)": [0.0],
            "Basis_converted sum (quantity < 0)": [0.0],
            "Comm/Fee_converted sum (quantity < 0)": [0.0]
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
                            "Comm/Fee_converted sum": 0.0,
                            # Dodajemy nowe kolumny
                            "Proceeds sum (quantity < 0)": 0.0,
                            "Proceeds_converted sum (quantity < 0)": 0.0,
                            "Comm/Fee sum (quantity < 0)": 0.0,
                            "Basis sum (quantity < 0)": 0.0, 
                            "Basis_converted sum (quantity < 0)": 0.0,
                            "Comm/Fee_converted sum (quantity < 0)": 0.0
                        }
                    all_years_summary[year]["Total_Sold"] += row["Total_Sold"]
                    all_years_summary[year]["Proceeds sum"] += row["Proceeds sum"]
                    all_years_summary[year]["Proceeds_converted sum"] += row["Proceeds_converted sum"]
                    all_years_summary[year]["Comm/Fee sum"] += row["Comm/Fee sum"]
                    all_years_summary[year]["Basis sum"] += row["Basis sum"]
                    all_years_summary[year]["Basis_converted sum"] += row["Basis_converted sum"]
                    all_years_summary[year]["Comm/Fee_converted sum"] += row["Comm/Fee_converted sum"]
                    # Dodajemy aktualizację nowych kolumn
                    all_years_summary[year]["Proceeds sum (quantity < 0)"] += row["Proceeds sum (quantity < 0)"]
                    all_years_summary[year]["Proceeds_converted sum (quantity < 0)"] += row["Proceeds_converted sum (quantity < 0)"]
                    all_years_summary[year]["Comm/Fee sum (quantity < 0)"] += row["Comm/Fee sum (quantity < 0)"]
                    all_years_summary[year]["Basis sum (quantity < 0)"] += row["Basis sum (quantity < 0)"]
                    all_years_summary[year]["Basis_converted sum (quantity < 0)"] += row["Basis_converted sum (quantity < 0)"]
                    all_years_summary[year]["Comm/Fee_converted sum (quantity < 0)"] += row["Comm/Fee_converted sum (quantity < 0)"]
            
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
                # Dodajemy aktualizację nowych kolumn
                all_summary["Proceeds sum (quantity < 0)"] += summary["Proceeds sum (quantity < 0)"].values[0]
                all_summary["Proceeds_converted sum (quantity < 0)"] += summary["Proceeds_converted sum (quantity < 0)"].values[0]
                all_summary["Comm/Fee sum (quantity < 0)"] += summary["Comm/Fee sum (quantity < 0)"].values[0]
                all_summary["Basis sum (quantity < 0)"] += summary["Basis sum (quantity < 0)"].values[0]
                all_summary["Basis_converted sum (quantity < 0)"] += summary["Basis_converted sum (quantity < 0)"].values[0]
                all_summary["Comm/Fee_converted sum (quantity < 0)"] += summary["Comm/Fee_converted sum (quantity < 0)"].values[0]
        
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
            "Comm/Fee_converted sum": [all_summary["Comm/Fee_converted sum"].values[0]],
            # Dodajemy nowe kolumny
            "Proceeds_converted sum (quantity < 0)": [all_summary["Proceeds_converted sum (quantity < 0)"].values[0]],
            "Basis_converted sum (quantity < 0)": [all_summary["Basis_converted sum (quantity < 0)"].values[0]],
            "Comm/Fee_converted sum (quantity < 0)": [all_summary["Comm/Fee_converted sum (quantity < 0)"].values[0]]
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
        columns=["id", "waluty", "Stock", "Date/Time", "Quantity", "Proceeds", "Comm/Fee", "Basis", "shares_in_possession"]
    )
    next_transaction_id = 1
    # Resetowanie do domyślnego pliku z kursami
    if exchange_rates_file != "kursy.csv" and os.path.exists(exchange_rates_file):
        if exchange_rates_file.startswith("uploaded_"):
            os.remove(exchange_rates_file)
    exchange_rates_file = "kursy.csv"
    return redirect(url_for("index"))


@app.route("/export-csv")
def export_csv():
    """
    Eksportuje dane do pliku CSV. W wierszach są akcje, a w kolumnach wartości:
    Proceeds sum, Proceeds_converted sum, Comm/Fee sum, Comm/Fee_converted sum
    oraz dodatkowe kolumny zawierające sumy dla transakcji sprzedaży.
    
    Dane są rozdzielone na lata - każda akcja ma wiersz podsumowujący oraz osobne wiersze dla każdego roku
    """
    global all_trades_df
    if all_trades_df.empty:
        return "Brak danych do eksportu", 400
    
    # Przetwarzamy dane jak w głównej funkcji
    processed_df = process_all_trades()
    
    if processed_df.empty:
        return "Brak przetworzonych danych do eksportu", 400
    
    # Tworzymy DataFrame dla eksportu, zawierający podsumowania dla każdej akcji
    export_data = []
    
    # Słownik do przechowywania sum dla podsumowania "All"
    all_yearly_data = {}
    
    for stock, group in processed_df.groupby("Stock"):
        # Dodajemy wiersz z ogólnym podsumowaniem dla danej akcji
        summary = summarize_transactions(group)
        if not summary.empty:
            export_data.append({
                "Stock": stock,
                "Year": "Total",  # Oznaczamy, że to wiersz z całkowitą sumą
                "Proceeds sum": summary["Proceeds sum"].values[0],
                "Proceeds_converted sum": summary["Proceeds_converted sum"].values[0],
                "Comm/Fee sum": summary["Comm/Fee sum"].values[0],
                "Comm/Fee_converted sum": summary["Comm/Fee_converted sum"].values[0],
                # Dodajemy nowe kolumny
                "Proceeds sum (quantity < 0)": summary["Proceeds sum (quantity < 0)"].values[0],
                "Proceeds_converted sum (quantity < 0)": summary["Proceeds_converted sum (quantity < 0)"].values[0],
                "Comm/Fee sum (quantity < 0)": summary["Comm/Fee sum (quantity < 0)"].values[0],
                "Basis sum (quantity < 0)": summary["Basis sum (quantity < 0)"].values[0],
                "Basis_converted sum (quantity < 0)": summary["Basis_converted sum (quantity < 0)"].values[0],
                "Comm/Fee_converted sum (quantity < 0)": summary["Comm/Fee_converted sum (quantity < 0)"].values[0]
            })
        
        # Dodajemy wiersze z podsumowaniem dla każdego roku
        yearly_summary = summarize_transactions_by_year(group)
        if not yearly_summary.empty:
            for _, year_row in yearly_summary.iterrows():
                year = year_row["Year"]
                
                # Dodajemy dane roczne do eksportu
                export_data.append({
                    "Stock": stock,
                    "Year": year,
                    "Proceeds sum": year_row["Proceeds sum"],
                    "Proceeds_converted sum": year_row["Proceeds_converted sum"],
                    "Comm/Fee sum": year_row["Comm/Fee sum"],
                    "Comm/Fee_converted sum": year_row["Comm/Fee_converted sum"],
                    # Dodajemy nowe kolumny
                    "Proceeds sum (quantity < 0)": year_row["Proceeds sum (quantity < 0)"],
                    "Proceeds_converted sum (quantity < 0)": year_row["Proceeds_converted sum (quantity < 0)"],
                    "Comm/Fee sum (quantity < 0)": year_row["Comm/Fee sum (quantity < 0)"],
                    "Basis sum (quantity < 0)": year_row["Basis sum (quantity < 0)"],
                    "Basis_converted sum (quantity < 0)": year_row["Basis_converted sum (quantity < 0)"],
                    "Comm/Fee_converted sum (quantity < 0)": year_row["Comm/Fee_converted sum (quantity < 0)"]
                })
                
                # Aktualizujemy sumy dla "All" w podziale na lata
                if year not in all_yearly_data:
                    all_yearly_data[year] = {
                        "Proceeds sum": 0,
                        "Proceeds_converted sum": 0,
                        "Comm/Fee sum": 0,
                        "Comm/Fee_converted sum": 0,
                        # Dodajemy nowe kolumny
                        "Proceeds sum (quantity < 0)": 0,
                        "Proceeds_converted sum (quantity < 0)": 0,
                        "Comm/Fee sum (quantity < 0)": 0,
                        "Basis sum (quantity < 0)": 0,
                        "Basis_converted sum (quantity < 0)": 0,
                        "Comm/Fee_converted sum (quantity < 0)": 0
                    }
                all_yearly_data[year]["Proceeds sum"] += year_row["Proceeds sum"]
                all_yearly_data[year]["Proceeds_converted sum"] += year_row["Proceeds_converted sum"]
                all_yearly_data[year]["Comm/Fee sum"] += year_row["Comm/Fee sum"]
                all_yearly_data[year]["Comm/Fee_converted sum"] += year_row["Comm/Fee_converted sum"]
                # Dodajemy aktualizację nowych kolumn
                all_yearly_data[year]["Proceeds sum (quantity < 0)"] += year_row["Proceeds sum (quantity < 0)"]
                all_yearly_data[year]["Proceeds_converted sum (quantity < 0)"] += year_row["Proceeds_converted sum (quantity < 0)"]
                all_yearly_data[year]["Comm/Fee sum (quantity < 0)"] += year_row["Comm/Fee sum (quantity < 0)"]
                all_yearly_data[year]["Basis sum (quantity < 0)"] += year_row["Basis sum (quantity < 0)"]
                all_yearly_data[year]["Basis_converted sum (quantity < 0)"] += year_row["Basis_converted sum (quantity < 0)"]
                all_yearly_data[year]["Comm/Fee_converted sum (quantity < 0)"] += year_row["Comm/Fee_converted sum (quantity < 0)"]
    
    # Dodajemy wiersz "All" z sumą wszystkich wartości
    all_row = {
        "Stock": "All",
        "Year": "Total",
        "Proceeds sum": sum(row["Proceeds sum"] for row in export_data if row["Year"] == "Total"),
        "Proceeds_converted sum": sum(row["Proceeds_converted sum"] for row in export_data if row["Year"] == "Total"),
        "Comm/Fee sum": sum(row["Comm/Fee sum"] for row in export_data if row["Year"] == "Total"),
        "Comm/Fee_converted sum": sum(row["Comm/Fee_converted sum"] for row in export_data if row["Year"] == "Total"),
        # Dodajemy nowe kolumny
        "Proceeds sum (quantity < 0)": sum(row["Proceeds sum (quantity < 0)"] for row in export_data if row["Year"] == "Total"),
        "Proceeds_converted sum (quantity < 0)": sum(row["Proceeds_converted sum (quantity < 0)"] for row in export_data if row["Year"] == "Total"),
        "Comm/Fee sum (quantity < 0)": sum(row["Comm/Fee sum (quantity < 0)"] for row in export_data if row["Year"] == "Total"),
        "Basis sum (quantity < 0)": sum(row["Basis sum (quantity < 0)"] for row in export_data if row["Year"] == "Total"),
        "Basis_converted sum (quantity < 0)": sum(row["Basis_converted sum (quantity < 0)"] for row in export_data if row["Year"] == "Total"),
        "Comm/Fee_converted sum (quantity < 0)": sum(row["Comm/Fee_converted sum (quantity < 0)"] for row in export_data if row["Year"] == "Total")
    }
    export_data.append(all_row)
    
    # Dodajemy wiersze "All" dla każdego roku
    for year, year_data in sorted(all_yearly_data.items()):
        export_data.append({
            "Stock": "All",
            "Year": year,
            "Proceeds sum": year_data["Proceeds sum"],
            "Proceeds_converted sum": year_data["Proceeds_converted sum"],
            "Comm/Fee sum": year_data["Comm/Fee sum"],
            "Comm/Fee_converted sum": year_data["Comm/Fee_converted sum"],
            # Dodajemy nowe kolumny
            "Proceeds sum (quantity < 0)": year_data["Proceeds sum (quantity < 0)"],
            "Proceeds_converted sum (quantity < 0)": year_data["Proceeds_converted sum (quantity < 0)"],
            "Comm/Fee sum (quantity < 0)": year_data["Comm/Fee sum (quantity < 0)"],
            "Basis sum (quantity < 0)": year_data["Basis sum (quantity < 0)"],
            "Basis_converted sum (quantity < 0)": year_data["Basis_converted sum (quantity < 0)"],
            "Comm/Fee_converted sum (quantity < 0)": year_data["Comm/Fee_converted sum (quantity < 0)"]
        })
    
    # Tworzymy DataFrame z przygotowanych danych
    export_df = pd.DataFrame(export_data)
    
    # Sortujemy dane dla lepszej czytelności (najpierw po Stock, potem po Year)
    export_df = export_df.sort_values(["Stock", "Year"], key=lambda x: x.map({"Total": 0}).fillna(x))
    
    # Zapisujemy do pamięci zamiast do pliku
    output = io.StringIO()
    export_df.to_csv(output, index=False)
    output.seek(0)
    
    # Generujemy nazwę pliku z datą
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trades_export_{timestamp}.csv"
    
    # Zwracamy plik do pobrania - z parametrami dla Flask 2.0+
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )


if __name__ == "__main__":
    app.run(debug=True)
