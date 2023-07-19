import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

df_ = pd.read_csv("flo_data_20K.csv")
df = df_.copy()

# Veriyi Anlama ve Hazırlama


def check_df(dataframe, head=5, quantiles=(0.05, 0.50, 0.95, 0.99, 1)):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Index ####################")
    print(dataframe.index)
    print("##################### Quantiles #####################")
    print(dataframe.describe(list(quantiles)).T)


check_df(df)

# Aykırı değerlerin baskılanması


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit)


for column in ["order_num_total_ever_online", "order_num_total_ever_offline",
               "customer_value_total_ever_offline", "customer_value_total_ever_online"]:
    replace_with_thresholds(df, column)

# Gerekli değişkenlerin hazırlanması
df["total_transaction"] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df["total_price"] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']

# Tarih belirten object veri tipindeki değişkenlerin veri tipinin datetime64[ns] olarak değiştirilmesi.
for column in df.loc[:, "first_order_date":"last_order_date_offline"].columns:
    df[column] = pd.to_datetime(df[column])

# BG/NBD ve Gamma-Gamma Modelleri için CLTV metriklerinin tanımlanması.

today_date = dt.datetime(2021, 6, 1)
cltv_df = pd.DataFrame({"master_id": df["master_id"],
                        "recency_cltv_weekly": (df["last_order_date"] - df["first_order_date"]).dt.days / 7,
                        "T_weekly": (today_date - df["first_order_date"]).apply(lambda x: x.days) / 7,
                        "frequency": df.loc[df["total_transaction"] > 1, "total_transaction"],
                        "monetary_cltv_avg": df["total_price"] / df["total_transaction"]})

# BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'], cltv_df['recency_cltv_weekly'], cltv_df['T_weekly'])

# 6 ay içerisinde müşterilerden beklenen satın sayısı
cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 6,
                                                                                       cltv_df['frequency'],
                                                                                       cltv_df['recency_cltv_weekly'],
                                                                                       cltv_df['T_weekly'])

# Müşterilerin ortalama bırakacakları değer tahmini
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])

# 6 aylık CLTV hesabı
cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                              cltv_df['frequency'],
                                              cltv_df['recency_cltv_weekly'],
                                              cltv_df['T_weekly'],
                                              cltv_df['monetary_cltv_avg'],
                                              time=6,
                                              freq="W",
                                              discount_rate=0.01)

# CLTV Değerine Göre Segmentlerin Oluşturulması
cltv_df["segment"] = pd.qcut(cltv_df["cltv"], [0, .4, .8, 1.], labels=["C", "B", "A"])

# Segmentlerin İncelenmesi
cltv_df.groupby("segment").agg({"cltv": ["mean", "min", "max"],
                                "frequency": ["mean", "min", "max", "sum"],
                                "monetary_cltv_avg": ["mean", "min", "max", "sum"],
                                "recency_cltv_weekly": ["mean", "min", "max"]})













