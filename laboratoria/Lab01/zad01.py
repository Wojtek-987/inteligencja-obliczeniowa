# czas spedzony nad programem: ~30min

import datetime
import math

# dane
name = input("Podaj imie: ")
year = input("Podaj rok urodzenia: ")
month = input("Podaj miesiac urodzenia: ")
day = input("Podaj dzien urodzenia: ")

# daty
data_ur = datetime.date(int(year), int(month), int(day))
data_dzis = datetime.date.today()
diff = data_dzis - data_ur

# wzory
y_p = math.sin((2 * math.pi) / 23 * diff.days)
y_e = math.sin((2 * math.pi) / 28 * diff.days)
y_i = math.sin((2 * math.pi) / 33 * diff.days)

print(f"Witaj, {name}! Urodziles(as) sie {diff.days} dni temu.")

print(f"Twoje wskazniki dzis to:\nFizyczny: {y_p}\nEmocjonalny: {y_e}\nIntelektualny: {y_i}")

if y_p >= .5 and y_e >= .5 and y_i >= .5:
    print("Swietny wynik!")
elif y_p < .5 and y_e < .5 and y_i < .5:
    print("Nie martw sie. Jutro bedzie lepiej!")
