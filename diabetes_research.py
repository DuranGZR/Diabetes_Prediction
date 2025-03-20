# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning
# 6. Prediction for a New Observation
# 7. Pipeline Main Function
from statistics import quantiles

import joblib
import pandas as pd
import seaborn as sns
from dask.array import blockwise
from matplotlib import pyplot as plt
from pandas.core.common import random_state
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from streamlit import dataframe

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

###################################################################
#        1. Exploratory Data Analysis (Keşifci Veri Analizi)      #
###################################################################

def check_df(dataframe, head=5):
    """
    DataFrame hakkında genel bilgi veren fonksiyon.

    Parametreler:
    dataframe : pd.DataFrame
        Analiz edilecek pandas DataFrame'i.
    head : int, opsiyonel (varsayılan=5)
        İlk ve son kaç satırın görüntüleneceğini belirler.

    Çıktılar:
    - DataFrame'in şekli (satır, sütun sayısı)
    - Sütun veri tipleri
    - İlk 'head' kadar satır
    - Son 'head' kadar satır
    - Eksik değer sayıları
    - Çeyrek değerler (min, 5%, 50%, 75%, 95%, 99%, max)
    """

    print("################## Shape ###################")
    print(dataframe.shape)  # DataFrame'in satır ve sütun sayısını yazdırır.

    print("################## Types ###################")
    print(dataframe.dtypes)  # Sütunlardaki veri tiplerini gösterir.

    print("################## Head ###################")
    print(dataframe.head(head))  # İlk 'head' kadar satırı yazdırır.

    print("################## Tail ###################")
    print(dataframe.tail(head))  # Son 'head' kadar satırı yazdırır.

    print("################## NA ###################")
    print(dataframe.isnull().sum())  # Eksik (NaN) değerleri sütun bazında hesaplar.

    print("################## Quantiles ###################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    # DataFrame'deki sayısal değişkenlerin çeyrek değerlerini hesaplar.


def cat_summary(dataframe, col_name, plot=False):
    """
    Kategorik değişkenlerin özetini veren fonksiyon.

    Parametreler:
    dataframe : pd.DataFrame
        Analiz edilecek pandas DataFrame'i.
    col_name : str
        İncelenecek kategorik değişkenin adı.
    plot : bool, opsiyonel (varsayılan=False)
        True olarak ayarlanırsa, değişkenin dağılımını görselleştirir.

    Çıktılar:
    - Değişkenin her bir kategorisinin kaç kez tekrarlandığı
    - Değişkenin her bir kategorisinin toplam veri içindeki yüzdesi
    - Opsiyonel olarak değişkenin countplot grafiği
    """

    # Kategorik değişkenin sınıflarını ve oranlarını gösteren bir DataFrame oluşturulur.
    summary_df = pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),  # Her kategori kaç kere tekrar ediyor?
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)  # Yüzdelik oranları hesaplar.
    })

    print(summary_df)
    print("###################################")

    # Eğer plot=True ise, kategorik değişkenin dağılımı görselleştirilir.
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)  # Grafiği göstermeyi sağlar.


def num_summary(dataframe, numerical_col, plot=False):
    """
    Sayısal değişkenlerin özet istatistiklerini veren fonksiyon.

    Parametreler:
    dataframe : pd.DataFrame
        Analiz edilecek pandas DataFrame'i.
    numerical_col : str
        İncelenecek sayısal değişkenin adı.
    plot : bool, opsiyonel (varsayılan=False)
        True olarak ayarlanırsa, değişkenin histogram grafiğini çizer.

    Çıktılar:
    - Sayısal değişkenin temel istatistiksel özetini yazdırır.
    - Opsiyonel olarak değişkenin histogram grafiğini gösterir.
    """

    # Çeyrek değerlerin hesaplanması için belirlenen yüzdelikler
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    # Sayısal değişkenin temel istatistiklerini yazdırır
    print(dataframe[numerical_col].describe(quantiles).T)

    # Eğer plot=True ise, histogram grafiği çizilir.
    if plot:
        dataframe[numerical_col].hist(bins=20)  # Histogram oluştur
        plt.xlabel(numerical_col)  # X ekseni etiketi
        plt.title(numerical_col)  # Başlık
        plt.show(block=True)  # Grafiği ekrana getir

def target_summary_with_num(dataframe, target, numerical_col):
    """
    Sayısal değişkenlerin hedef değişkene göre ortalamasını hesaplayan fonksiyon.

    Parametreler:
    dataframe : pd.DataFrame
        Analiz edilecek pandas DataFrame'i.
    target : str
        Hedef değişkenin adı (genellikle kategorik değişken).
    numerical_col : str
        İncelenecek sayısal değişkenin adı.

    Çıktılar:
    - Hedef değişkenin her bir kategorisi için sayısal değişkenin ortalama değerlerini yazdırır.
    """

    # Hedef değişkene göre sayısal değişkenin ortalamasını hesaplar.
    summary = dataframe.groupby(target).agg({numerical_col: 'mean'})

    print(summary, end="\n\n\n")  # Sonuna birkaç satır boşluk ekleyerek okunabilirliği artırır.



def target_summary_with_cat(dataframe, target, categorical_col):
    """
    Kategorik değişkenin her bir sınıfı için hedef değişkenin ortalamasını hesaplayan fonksiyon.

    Parametreler:
    dataframe : pd.DataFrame
        Analiz edilecek pandas DataFrame'i.
    target : str
        Hedef değişkenin adı (genellikle sayısal değişken).
    categorical_col : str
        İncelenecek kategorik değişkenin adı.

    Çıktılar:
    - Kategorik değişkenin her bir sınıfı için hedef değişkenin ortalama değerlerini içeren DataFrame.
    """

    # Kategorik değişkenin her bir sınıfı için hedef değişkenin ortalamasını hesaplar.
    summary = pd.DataFrame({"TARGET MEAN": dataframe.groupby(categorical_col)[target].mean()})

    print(summary, end="\n\n\n")  # Sonuna boş satırlar ekleyerek okunabilirliği artırır.




def correlation_matrix(df, cols):
    """
    Belirtilen değişkenler için korelasyon matrisini hesaplayıp ısı haritası olarak görselleştiren fonksiyon.

    Parametreler:
    df : pd.DataFrame
        Analiz edilecek pandas DataFrame'i.
    cols : list
        Korelasyon matrisi hesaplanacak sütun isimlerini içeren liste.

    Çıktılar:
    - Korelasyon matrisini içeren ısı haritası (heatmap).
    """

    fig = plt.gcf()  # Mevcut grafik figürünü alır.
    fig.set_size_inches(10, 8)  # Grafik boyutunu belirler.

    plt.xticks(fontsize=10)  # X ekseni etiketlerinin font boyutunu ayarlar.
    plt.yticks(fontsize=10)  # Y ekseni etiketlerinin font boyutunu ayarlar.

    # Seaborn heatmap ile korelasyon matrisini görselleştirir.
    fig = sns.heatmap(df[cols].corr(),  # Korelasyon matrisini hesaplar.
                      annot=True,  # Her hücreye korelasyon değerini yazar.
                      linewidths=0.5,  # Hücreler arasındaki çizgi kalınlığı.
                      annot_kws={"size": 12},  # Anotasyon font boyutu.
                      linecolor="w",  # Hücreler arasındaki çizgi rengi (beyaz).
                      cmap="RdBu")  # Renk haritası (kırmızı-mavi).

    plt.show(block=True)  # Grafiği ekrana getirir.


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car







df = pd.read_csv("/datasets/diabetes.csv")


check_df(df)

# Değişken türlerinin ayrıştırılması

cat_cols, num_cols, cat_but_car = grab_col_names(df,cat_th=5,car_th=20)


# Kategorik değişkenlerin incelenmesi
for col in cat_cols:
    cat_summary(df,col)

# Sayısal değişkenlerin incelenmesi
df[num_cols].describe().T

#for col in num_cols:
#    num_summary(df,col,plot=True)

# Sayısal değişkenlerin birbirleri ile korelasyonu
correlation_matrix(df,num_cols)

# Target ile sayısal değişkenlerin incelenmesi
for col in num_cols:
    target_summary_with_num(df,"Outcome",col)


#########################################################################
#      2. Data Preprocessing & Feature Engineering (Veri ön işleme)     #
#########################################################################

def outlier_thresholds(dataframe, col_name, q1 = 0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    interquantile_range = quartile3 - quartile1

    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe




# Değişken isimleri büyütmek
df.columns = [col.upper() for col in df.columns]

# Glucose
df['NEW_GLUCOSE_CAT'] = pd.cut(x=df['GLUCOSE'], bins=[-1, 139, 200], labels=["normal", "prediabetes"])

# Age
df.loc[(df['AGE'] < 35), "NEW_AGE_CAT"] = 'young'
df.loc[(df['AGE'] >= 35) & (df['AGE'] <= 55), "NEW_AGE_CAT"] = 'middleage'
df.loc[(df['AGE'] > 55), "NEW_AGE_CAT"] = 'old'


# BMI
df['NEW_BMI_RANGE'] = pd.cut(x=df['BMI'], bins=[-1, 18.5, 24.9, 29.9, 100],
                             labels=["underweight", "healty", "overweight", "obese"])

# BloodPressure
df['NEW_BLOODPRESSURE'] = pd.cut(x=df['BLOODPRESSURE'], bins=[-1, 79, 89, 123], labels=["normal", "hs1", "hs2"])


check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)


for col in cat_cols:
    cat_summary(df, col)

for col in cat_cols:
    target_summary_with_cat(df, "OUTCOME", col)

cat_cols = [col for col in cat_cols if 'OUTCOME' not in col]


df = one_hot_encoder(df,cat_cols, drop_first=True)

check_df(df)


df.columns = [col.upper() for col in df.columns]



# Son güncel değişken türlerimi tutuyorum.

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

cat_cols = [col for col in cat_cols if 'OUTCOME' not in col]

for col in num_cols:
    print(col, check_outlier(df, col, q1=0.05, q3=0.95))

replace_with_thresholds(df,'INSULIN')

for col in num_cols:
    print(col, check_outlier(df, col, q1=0.05, q3=0.95))



# Standartlaştırma
X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)


y = df['OUTCOME']
X = df.drop('OUTCOME', axis=1)

check_df(X)



# FONSİYONLAŞTIRMA

def diabetes_data_prep(dataframe):
    dataframe.columns = [col.upper() for col in dataframe.columns]

    # Glucose kategorisi
    dataframe['NEW_GLUCOSE_CAT'] = pd.cut(x=dataframe['GLUCOSE'], bins=[-1, 139, 200], labels=["normal", "prediabetes"]).astype(str)

    # Age kategorisi
    dataframe.loc[(dataframe['AGE'] < 35), "NEW_AGE_CAT"] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 35) & (dataframe['AGE'] <= 55), "NEW_AGE_CAT"] = 'middleage'
    dataframe.loc[(dataframe['AGE'] > 55), "NEW_AGE_CAT"] = 'old'

    # BMI kategorisi
    dataframe['NEW_BMI_RANGE'] = pd.cut(x=dataframe['BMI'], bins=[-1, 18.5, 24.9, 29.9, 100],
                                        labels=["underweight", "healty", "overweight", "obese"]).astype(str)

    # Blood Pressure kategorisi
    dataframe['NEW_BLOODPRESSURE'] = pd.cut(x=dataframe['BLOODPRESSURE'], bins=[-1, 79, 89, 123],
                                            labels=["normal", "hs1", "hs2"]).astype(str)

    # Kategorik ve sayısal değişkenleri belirleme
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5, car_th=20)
    cat_cols = [col for col in cat_cols if 'OUTCOME' not in col]

    # One-hot encoding işlemi (kategorik değişkenler için)
    df = one_hot_encoder(dataframe, cat_cols, drop_first=True).astype(float)  # Float olarak saklıyoruz.

    df.columns = [col.upper() for col in df.columns]

    # Tekrar kategorik ve sayısal değişkenleri belirleme
    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
    cat_cols = [col for col in cat_cols if 'OUTCOME' not in col]

    # Eşik değerleri ile değiştirme (eğer float oluyorsa int'e çevrilebilir)
    replace_with_thresholds(df, 'INSULIN')
    df['INSULIN'] = df['INSULIN'].astype(int)  # Hata devam ederse kaldırılabilir.

    # Standardizasyon işlemi (num_cols için)
    X_scaled = StandardScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns, index=df.index)  # DataFrame formatında ekledik.

    # Bağımsız değişkenler (X) ve bağımlı değişken (y)
    y = df["OUTCOME"]
    X = df.drop(["OUTCOME"], axis=1)

    return X, y


df = pd.read_csv("/datasets/diabetes.csv")

check_df(df)

X, y = diabetes_data_prep(df)

check_df(X)



#####################################################
#               Base Models                         #
#####################################################

def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

base_models(X,y)


######################################################
#      Automated Hyperparameter Optimization         #
######################################################

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {
    'max_depth': range(1, 20),
    "min_samples_split": range(2, 30)
}

rf_params = {
    "max_depth": [8, 15, None],
    "max_features": [5, 7, "sqrt"],
    "min_samples_split": [15, 20],
    "n_estimators": [200, 300]
}

xgboost_params = {
    "learning_rate": [0.1, 0.01],
    "max_depth": [5, 8],
    "n_estimators": [100, 200]
}

lightgbm_params = {
    "learning_rate": [0.01, 0.1],
    "n_estimators": [300, 500]
}

classifiers = [
    ('KNN', KNeighborsClassifier(), knn_params),
    ("CART", DecisionTreeClassifier(), cart_params),
    ("RF", RandomForestClassifier(), rf_params),
    ('XGBoost', XGBClassifier(eval_metric='logloss'), xgboost_params),
    ('LightGBM', LGBMClassifier(), lightgbm_params)
]






def hyperparameter_optimization(X, y, cv=5, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y)


######################################################
#          Stacking & Ensemble Learning              #
######################################################


def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

voting_clf = voting_classifier(best_models, X, y)


######################################################
#       Prediction for a New Observation             #
######################################################


X.columns

random_user = X.sample(1,random_state=45 )

voting_clf.predict(random_user)


joblib.dump(voting_clf, "voting_clf2.pkl")

new_model = joblib.load("voting_clf2.pkl")
new_model.predict(random_user)

















