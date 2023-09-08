# author : Mustafa Erdem Gülsoy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = pd.read_csv("datasets/persona.csv")

# DataFrame için ilk olarak categoric ve numeric değişkenleri ayırıyoruz ;
def grab_col_names(dataframe, cat_th=5, car_th=20) :
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe : dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th : int, float
        numerik fakat kategorik olan değişkenlerdir.
    car_th : int, float
        kategorik fakat kardinal olan değişkenler için sınıf eşik değeri.

    Returns
    -------
        cat_cols : list
            Kategorik değişken listesi.
        num_cols : list
            Numerik değişkeen listesi.
        cat_but_car : list
            Kategorik görünümlü kardinal değişken listesi.

    Notes
    ------
        cat_cols + num_cols +cat_but_car = toplam değişken sayısı.
        num_but_cat, cat_cols'un içerisinde.

    """

    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].dtypes in ["int64", "float64"] and df[col].nunique() < cat_th]

    cat_but_car = [col for col in df.columns if str(df[col].dtypes) in ["category", "object"] and df[col].nunique() > car_th]

    cat_cols += num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if col not in cat_cols and df[col].dtypes in ["int64", "float64"]]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# DataFrame için ön bilgi alma işlemi ;
def check_df (dataframe, head=5) :
    print("#################### Shape ##################")
    print(dataframe.shape)
    print("#################### Type ##################")
    print(dataframe.dtypes)
    print("#################### Head ##################")
    print(dataframe.head(head))
    print("#################### Tail ##################")
    print(dataframe.tail(head))
    print("#################### NA ##################")
    print(dataframe.isnull().sum())
    print("#################### Quantiles ##################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)
def cat_summary2(dataframe, col_name, plot=False) :
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio"  : 100 * dataframe[col_name].value_counts() / len(dataframe) }))
    print("###############################################")

    if plot :
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols :
    cat_summary2(df, col, True)


def num_summary2(dataframe, numerical_col, plot=False) :
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot :
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols :
    num_summary2(df, col, True)

# Veri setindeki değişkenlerin nunique ve frekans sayıları ;
for col in df.columns :
    print(col, df[col].nunique(),"\n")
    print(df[col].value_counts(),"\n")

# Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby("COUNTRY")["PRICE"].sum() # veya ;
df.groupby('COUNTRY').agg({'PRICE': 'sum'})

# Ülkelere göre PRICE ortalamaları nedir?
df.groupby("COUNTRY")["PRICE"].mean() # veya ;
df.groupby('COUNTRY').agg({'PRICE': 'mean'})

# SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby("SOURCE")["PRICE"].mean()

# COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(["COUNTRY", "SOURCE"])["PRICE"].mean().reset_index()

# COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])["PRICE"].mean()
# Burada işlemi yaptık ancak burada price dışında tüm çıktılar index ismidir.

# Çıktıyı PRICE'a göre azalan şekilde sıralayalım ve indexte yer alan isimleri değişken ismine çevirelim ;
agg_df = agg_df.sort_values(ascending=False).reset_index()

# Age değişkenini kategorik değişkene çevirelim ve agg_df’e ekleyelim ;
range = [0, 18, 24, 31, 41, 70] # aralıkların başlangıçlarını yazıyoruz.
etiket = ["0-18", "19-23", "24-30", "31-40", "41-70"] # aralıkların isimlendirilmesi.
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=range, right=False, labels=etiket)

# Tüm bilgileri içeren ve aralığını gösteren bir değişken oluşturup agg_df içine ekleyelim ;
new_agg = agg_df.drop(['AGE', 'PRICE'], axis=1)
agg_df["customers_level_based"] = ["_".join(col).upper() for col in new_agg.values]

# Gereksiz değişkenleri çıkaralım ;
agg_df = agg_df[['customers_level_based', 'PRICE']]
agg_df = agg_df.groupby('customers_level_based')['PRICE'].mean().reset_index()

# Yeni müşterileri (personaları) price'a göre segmentlere (4) ayıralım ;
agg_df['SEGMENT'] = pd.qcut(agg_df["PRICE"], q=4, labels=['D', 'C', 'B','A'])

# Segmentleri betimleyelim. (Segmentlere göre groupby yapıp price mean, max, sum’larını alalım) ;
agg_df.groupby("SEGMENT")["PRICE"].mean()

