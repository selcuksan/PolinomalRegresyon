import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#2 VERİ ÖNİŞLEME
#2.1 VERİ YÜKLEME
df = pd.read_csv("maaslar.csv")
print(df.head())
print()

x = df[["Egitim Seviyesi"]]
y = df[["maas"]]
#LİNEER REGRESYON YÖNTEMİ
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(x,y)
plt.scatter(x,y,c="r")
plt.plot(x,lin_reg.predict(x),label="degree 1")

#POLINOMAL REGRESYON YÖNTEMİ

from sklearn.preprocessing import PolynomialFeatures

pol_reg = PolynomialFeatures(degree=2)
X_pol=pol_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_pol,y)
pol_reg_tahmin = lin_reg2.predict(X_pol)
plt.plot(x,pol_reg_tahmin,label="degree 2")

pol_reg2 = PolynomialFeatures(degree=4)
X_pol=pol_reg2.fit_transform(x)
lin_reg3 = LinearRegression()
lin_reg3.fit(X_pol,y)
pol_reg_tahmin = lin_reg3.predict(X_pol)
plt.plot(x,pol_reg_tahmin,label="degree 4")

print(f"gercek deger: {y.iloc[9:10,0:1].values}")
print(f"degree 1 tahmin: {lin_reg.predict([[10]])}")
print(f"degree 2 tahmin: {lin_reg2.predict(pol_reg.fit_transform([[10]]))}")
print(f"degree 4 tahmin: {lin_reg3.predict(pol_reg2.fit_transform([[10]]))}")
plt.legend()
plt.show()
