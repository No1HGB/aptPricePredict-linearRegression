import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

cdrate = pd.read_excel('./data/cdrate.xlsx')
amount = pd.read_excel('./data/amount.xlsx')
trade = pd.read_excel('./data/trade.xlsx')
price = pd.read_excel('./data/price.xlsx')

seoulData = pd.concat([cdrate.date, cdrate.cdrate, amount.seoul, trade.seoul, price.seoul],
                      axis=1, keys=["date", "cdrate", "amount", "trade", "price"])
busanData = pd.concat([cdrate.date, cdrate.cdrate, amount.busan, trade.busan, price.busan],
                      axis=1, keys=["date", "cdrate", "amount", "trade", "price"])
daeguData = pd.concat([cdrate.date, cdrate.cdrate, amount.daegu, trade.daegu, price.daegu],
                      axis=1, keys=["date", "cdrate", "amount", "trade", "price"])
incheonData = pd.concat([cdrate.date, cdrate.cdrate, amount.incheon, trade.incheon, price.incheon],
                      axis=1, keys=["date", "cdrate", "amount", "trade", "price"])
gwangjuData = pd.concat([cdrate.date, cdrate.cdrate, amount.gwangju, trade.gwangju, price.gwangju],
                      axis=1, keys=["date", "cdrate", "amount", "trade", "price"])
daejeonData = pd.concat([cdrate.date, cdrate.cdrate, amount.daejeon, trade.daejeon, price.daejeon],
                      axis=1, keys=["date", "cdrate", "amount", "trade", "price"])
ulsanData = pd.concat([cdrate.date, cdrate.cdrate, amount.ulsan, trade.ulsan, price.ulsan],
                      axis=1, keys=["date", "cdrate", "amount", "trade", "price"])

#최종 데이터(dataframe)
seoul = seoulData.set_index(keys=['date'], inplace=False, drop=True)
busan = busanData.set_index(keys=['date'], inplace=False, drop=True)
daegu = daeguData.set_index(keys=['date'], inplace=False, drop=True)
incheon = incheonData.set_index(keys=['date'], inplace=False, drop=True)
gwangju = gwangjuData.set_index(keys=['date'], inplace=False, drop=True)
daejeon = daejeonData.set_index(keys=['date'], inplace=False, drop=True)
ulsan = ulsanData.set_index(keys=['date'], inplace=False, drop=True)

#상관관계 파악(상관관계매트릭스)
seoul_corr_matrix = seoul.corr()
busan_corr_matrix = busan.corr()
daegu_corr_matrix = daegu.corr()
incheon_corr_matrix = incheon.corr()
gwangju_corr_matrix = gwangju.corr()
daejeon_corr_matrix = daejeon.corr()
ulsan_corr_matrix = ulsan.corr()

seoul_corr = seoul_corr_matrix["price"].sort_values(ascending=False)
busan_corr = busan_corr_matrix["price"].sort_values(ascending=False)
daegu_corr = daegu_corr_matrix["price"].sort_values(ascending=False)
incheon_corr = incheon_corr_matrix["price"].sort_values(ascending=False)
gwangju_corr = gwangju_corr_matrix["price"].sort_values(ascending=False)
daejeon_corr = daejeon_corr_matrix["price"].sort_values(ascending=False)
ulsan_corr = ulsan_corr_matrix["price"].sort_values(ascending=False)

print( seoul_corr, busan_corr, daegu_corr, incheon_corr, gwangju_corr, daejeon_corr, ulsan_corr)

#train,test 분리
seoulX = seoul[["cdrate", "amount", "trade"]]
seoulY = seoul[["price"]]
busanX = busan[["cdrate", "amount", "trade"]]
busanY = busan[["price"]]
daeguX = daegu[["cdrate", "amount", "trade"]]
daeguY = daegu[["price"]]
incheonX = incheon[["cdrate", "amount", "trade"]]
incheonY = incheon[["price"]]
gwangjuX = gwangju[["cdrate", "amount", "trade"]]
gwangjuY = gwangju[["price"]]
daejeonX = daejeon[["cdrate", "amount", "trade"]]
daejeonY = daejeon[["price"]]
ulsanX = ulsan[["cdrate", "amount", "trade"]]
ulsanY = ulsan[["price"]]

seoulX_train, seoulX_test, seoulY_train, seoulY_test = train_test_split(seoulX, seoulY, train_size=0.7, test_size=0.3)
busanX_train, busanX_test, busanY_train, busanY_test = train_test_split(busanX, busanY, train_size=0.7, test_size=0.3)
daeguX_train, daeguX_test, daeguY_train, daeguY_test = train_test_split(daeguX, daeguY, train_size=0.7, test_size=0.3)
incheonX_train, incheonX_test, incheonY_train, incheonY_test = train_test_split(incheonX, incheonY,
                                                                                train_size=0.7, test_size=0.3)
gwangjuX_train, gwangjuX_test, gwangjuY_train, gwangjuY_test = train_test_split(gwangjuX, gwangjuY,
                                                                                train_size=0.7, test_size=0.3)
daejeonX_train, daejeonX_test, daejeonY_train, daejeonY_test = train_test_split(daejeonX, daejeonY,
                                                                                train_size=0.7, test_size=0.3)
ulsanX_train, ulsanX_test, ulsanY_train, ulsanY_test = train_test_split(ulsanX, ulsanY, train_size=0.7, test_size=0.3)

#다중선형회귀 모델링 & 예측
mlr = LinearRegression()

#도시 예측, 실제론 모델 fitting시 하나씩만 fitting
mlr.fit(seoulX_train.values, seoulY_train.values)
print(mlr.score(seoulX_train.values, seoulY_train.values))
print(mlr.predict([[1.94, 3969, 1377.9]]))

mlr.fit(busanX_train.values, busanY_train.values)
print(mlr.score(busanX_train.values, busanY_train.values))
print(mlr.predict([[1.94, 2972, 1713.2]]))

mlr.fit(daeguX_train.values, daeguY_train.values)
print(mlr.score(daeguX_train.values, daeguY_train.values))
print(mlr.predict([[1.94, 2318, 1252.4]]))

mlr.fit(incheonX_train.values, incheonY_train.values)
print(mlr.score(incheonX_train.values, incheonY_train.values))
print(mlr.predict([[1.94, 2717, 1336.1]]))

mlr.fit(gwangjuX_train.values, gwangjuY_train.values)
print(mlr.score(gwangjuX_train.values, gwangjuY_train.values))
print(mlr.predict([[1.94, 1176, 1568.3]]))

mlr.fit(daejeonX_train.values, daejeonY_train.values)
print(mlr.score(daejeonX_train.values, daejeonY_train.values))
print(mlr.predict([[1.94, 1937, 766.5]]))

mlr.fit(ulsanX_train.values, ulsanY_train.values)
print(mlr.score(ulsanX_train.values, ulsanY_train.values))
print(mlr.predict([[1.94, 570, 829.7]]))