# proje için gerekli kütüphaneleri dosyama import ettim:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import numpy as np

dr=pd.read_csv("home_price.csv") # pandas kütüphanesindeki read_csv komutunu kullanarak kaggle'dan aldığım csv formatındaki veri setini okuttum ve bunu dr (datarahmen) adlı değişkene atadım. 
print(dr.head()) # dr adlı değişkende verisetim vardı bu veri setindeki ilk 5 satırı görmem açısından bu head komutunu kullandım ve ekrana yazdırttım ki görebileyim.
print(dr.info()) # veri setim hakkındaki bilgileri (kolonlardaki dolu veri sayılarını görmem açısından çünkü boş olan verileri bu sayısal değerlere göre doldurma yapacağım veya yapmayacağım) görmek açısından info komutunu kullanarak ekrana yazdırttım.
dr=dr.drop(columns=["Tapu_Durumu"], inplace=False) # info komutu ile Tapu_Durumu adlı kolondaki eksik veri çok fazlaydı ve bu yüzden bu kolonda doldurma işlemi yapmam modelin yanlış eğitilmesine neden olurdu bu sebeple bu alanı silmem daha doğru olur diye düşünerek bu alanı sildim. 
dr['Oda_Sayısı']=dr["Oda_Sayısı"].fillna(dr["Oda_Sayısı"].median()) # info komutu ile elde ettiğim bilgiler doğrultusunda Oda_Sayısı adlı kolonda az miktarda veri eksikliği/boş veri vardı. Bu kolon (Oda_Sayısı) hedef değişkene (Fiyat) oldukça fazla etki ettiğinden bu kolonu silemem. Zaten eksik verisi de çok yok. Bu sebeple bu alanı doldurmam gerekir. Doldurma işlemi için fillna komutu kullanılır. medyana göre doldurma işlemi yaptırdım çünkü bu kolonun içindeki veriler integer veri tipindeydi. Sayısal bir veri tipi olduğundan medyana göre doldurma işlemi yaptırdım.
dr['Banyo_Sayısı']=dr["Banyo_Sayısı"].fillna(dr["Banyo_Sayısı"].median()) # aynı şekilde info komutunda Banyo_Sayısı adlı alanın da boş verisi az sayıdaydı ve bu alanın içindeki veriler de sayısal türdeydi. Bu sebeple bu alana da medyana göre doldurma işlemi yaptırdım.
dr['Bulunduğu_Kat']=dr["Bulunduğu_Kat"].fillna(dr["Bulunduğu_Kat"].mode()[0]) # yine info komutunda Bulunduğu_Kat alanındaki boş veri sayısı da az olduğundan bu alanı da doldurttum fakat bu alanın içindeki veriler string türdeydi. string türdeki verilerde doldurma işlemi moda göre yapılır diye biliyorum bu sebeple moda göre doldurma işlemi yaptırdım. Tabi birden fazla mod değeri gelebilir bu sebeple 0.indeksteki yani ilk mod değerini alması gerektiğini belirttim ki karışıklık olmasın. 
# Bulunduğu_Kat adlı alanda yaptıklarımı aynı sebeplerle Eşya_Durumu, Yatırıma_Uygunluk, Takas adlı alanlara da yaptırdım:
dr['Eşya_Durumu'] = dr['Eşya_Durumu'].fillna(dr['Eşya_Durumu'].mode()[0]) 
dr['Yatırıma_Uygunluk'] = dr['Yatırıma_Uygunluk'].fillna(dr['Yatırıma_Uygunluk'].mode()[0])
dr['Takas'] = dr['Takas'].fillna(dr['Takas'].mode()[0])
print(dr.info()) # doldurma ve silme işlemlerinden sonra son durumu görmek açısından tekrar info ile veri setindeki bilgileri alıp ekrana yazdırttım.
print(f"istatistikler:{dr.describe()}") # ilerleyen aşamalarda sayısal değerde verilere sahip alanların istatistiksel verileri (max ve min değerleri, birinci çeyrekliği, medyanı, 3.çeyrekliği vs.) lazım olacağından (bu istatistiksel veriler doğrultusunda veri temizlemeye gideceğim çünkü) bu komutu yazdım.
yeni_kategori=['Bulunduğu_Kat', 'Eşya_Durumu', 'Yatırıma_Uygunluk','Takas', 'Binanın_Yaşı', 'Isıtma_Tipi','Şehir','Kullanım_Durumu'] # veri tipleri string olan alanları yeni_kategori adlı listeye eklettim.
dr_coded=pd.get_dummies(dr, columns=yeni_kategori, drop_first=True) # model sayısal değerlerle çalışacağından veri tipleri string olan alanların verilerini sayısallaştırmam gerekir bu sayısallaştırma işlemini get_dummies komutu ile yaptığımızdan get_dummies fonksiyonunu yazdım. parametre olarak verisetinin olduğu değişken (dr) ile string verilerin olduğu alanların listesini (yeni_kategori) aldı. son durumu yani string verilerin de sayısallaştırıldığı son durumu dr_coded adlı değişkene atadım. Çoklu doğrusal bağlantı olmaması açısından drop_first=True komutunu yazdım.
print(dr_coded.info()) # son durumun istatistiksel verilerini görmek açısından tekrar ekrana yazdırdım.
# Fiyat alanındaki uç değerleri çıkarttım:
max=dr_coded['Fiyat'].quantile(0.005)  
min=dr_coded['Fiyat'].quantile(0.995) 
dr_coded=dr_coded[(dr_coded['Fiyat']>max) & (dr_coded['Fiyat']<min)]
print(f"Fiyat temizlik sonrası sayı: {len(dr_coded)}")
# Net_Metrekare alanındaki uç değerleri çıkarttım:
max_m2=dr_coded['Net_Metrekare'].quantile(0.005)
min_m2=dr_coded['Net_Metrekare'].quantile(0.995)
dr_coded=dr_coded[(dr_coded['Net_Metrekare']>max_m2) & (dr_coded['Net_Metrekare']<min_m2)]
# Brüt_Metrekare alanındaki uç değerleri çıkarttım:
max_m3=dr_coded['Brüt_Metrekare'].quantile(0.005)
min_m3=dr_coded['Brüt_Metrekare'].quantile(0.995)
dr_coded=dr_coded[(dr_coded['Brüt_Metrekare']>max_m3) & (dr_coded['Brüt_Metrekare']<min_m3)]
print(f" Fiyat, Net_Metrekare ve Brüt_Metrekare Temizlik sonrası sayı: {len(dr_coded)}")
# Oda_Sayısı alanındaki uç değerleri çıkarttım:
dr_coded=dr_coded[(dr_coded['Oda_Sayısı']>=1) & (dr_coded['Oda_Sayısı']<=8)]
print(f"En son tam temizlenmiş satır sayısı: {len(dr_coded)}")

# MODELİ EĞİTME:
x=dr_coded.drop('Fiyat', axis=1) # axis=1 sütun demektir. drop komutu ise silme işlemini yapar. Yani bu satır Fiyat adındaki alanı/sütunu siler ve Fiyat alanı silinmiş veri tabanının son halini x değişkenine atar.
y=dr_coded['Fiyat'] # dr_coded değişkeninde string verilerin sayısallaştırılmış son durumu tutuluyotdu. Bu veri setinden sadece Fiyat adlı alan/sütun alınır ve y değişkenine atanır. Amaç modelin hedef değişkeni (y) ile diğer özellikleri (x) ayırmak

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) #  Veri setini %80'i eğitim verisi %20si test verisi olacak şekilde ikiye ayırttım.
scaler=StandardScaler() # Ölçeklemek için gerekli fonksiyonu scaler adlı değişkene atadım ki değişkeni her kullandığımda bu fonksiyon çağırılsın. Verideki tüm sayıları aynı boyuta getirir, birinin diğerinden üstün olmasını engeller.
numeric_olanlar=['Net_Metrekare','Brüt_Metrekare', 'Oda_Sayısı', 'Binanın_Kat_Sayısı', 'Banyo_Sayısı'] # ölçeklemenin yapılacağı alanları (içerisinde sayısal verilerin olduğu alanları) numeric_olanlar adlı listede tuttum ki her numeric alan için kod tekrarına düşmeyeyim diye.

X_train[numeric_olanlar]=scaler.fit_transform(X_train[numeric_olanlar]) # x_trainde fit ve transform yapılır-> fit ile öğrenir, transform ile öğrendiğini x_train üzerinde uygular.
X_test[numeric_olanlar]=scaler.transform(X_test[numeric_olanlar]) # model tahmin yaparken test verilerine bakacak ve onlar standartilize edilmediğinden doğru anlayamayacak ve tahmini hatalı olacak. Bu yüzden de x_test'e de transform() uyguladık. Ayrıca x_test'te fit ile öğrenme yapılmaz yapılırsa kopya vermiş oluruz yani bilgi sızıntısı gerçekleşir. Bunun olmaması için x_test verilerinde fit komutu kullanılmaz. hedef değişkende de (y verilerinde-> y_test veya y_train) fit veya transform komutu yapılmaz çünkü hedef değişkende standartlaşmaya gitme gibi bir amacımız olamaz. Bize TL cinsinden fiyat değişkeninin verileri lazım :)

modell=LinearRegression() # Linear Regression fonksiyonunu modell adlı değişkene atadım böylece modell değişkeni her çağırıldığında aslında Linear Regression fonksiyonu çağırılmış olacak. Linear Regression, verilere en uygun düz çizgiyi çizerek bir tahmin denklemi oluşturmak için kullanılır.
modell.fit(X_train,y_train) # Modeli eğitmek için bu satırı yazdım. Yani model X ve Y arasındaki ilişkiyi öğrenir. ölçeklenmiş eğitim özellikleri (x_train) ile doğru cevaplar arasındaki ilişki öğrenilir. Böyle böyle olursa fiyat böyle böyle oluyor demesi gibi...
print(modell.coef_) # modell değişkeni artık eğitilmiş modeli tutuyor, bu eğitilmiş modelde özelliklerin katsayılarını ekrana yazdırtıp görmek için bu satırı yazdım.
tahminler=modell.predict(X_test) # model x_train ile öğrenmişti burada öğrendiklerini daha önce hiç görmediği x_test verilerine uygulayarak tahminde bulunur. x_test'in cevapları zaten veritabanında belli olduğu için bu aşamada modelin ne kadar doğru tahmin yaptığını görebiliriz. Bu yaptığı tahminleri de tahminler adlı değişkende tutuyoruz.
print(tahminler) # modelin tahmin ettiklerini ekrana yazdırdım karşılaştırma yapabilmek açısından.
dogruluk_oranı=r2_score(y_test,tahminler) # r2_score modelin ne kadar doğru tahmin ettiğini gösterir. Tahmin etme işlemi için 2 şey gerekir mantıken-> tahmin ettiği değerler (tahminler değişkeni) ile gerçek cevaplar/ doğru bulmasını beklediğimiz cevaplar(y_test değişkeni) r2_score fonksiyonu parametre olarak bu değişkenleri bu yüzden aldı yani... Sonuç olarak modelin ne kadar oranda doğru tahmin ettiğini görmek isteriz ona göre modelde iyileştirmeye gidilir çünkü. Bu sebeple r2_score değerini hesaplattım ve bu hesabın sonucunu da dogruluk_oranı adlı değişkene atadım.
print(dogruluk_oranı) # modelin ne kadar doğru tahmin ettiğini ekrana yazdırttım.
moh=mean_absolute_error(y_test,tahminler) # moh=mutlak ortalama hata-> modelin ortalama olarak ne kadar yanıldığını gösterir. Bunu da ekrana yazdırmamın nedeni yine modelin tahmin etme gücünde iyileştirmeye digip gidilmeyeceğine karar vermek için.
print(moh)
capraz_skor= cross_val_score(estimator=modell,X=x,y=y,cv=5,scoring='r2') # girilen cv değeri kadar veriyi eşit parçalara böler ve her parça için (her döngüde) O parçayı Test Seti olarak ayırır geriye kalan tüm parçaları Eğitim Seti olarak birleştirir. Eğitim seti üzerinde modeli yeniden eğitir. Test seti üzerinde tahmin yapar ve belirlenen ölçüm skorunu (r2) hesaplar-> parametre olarak r2 girdiğimden r2 türünde hesaplar. Tüm bu işlemleri yaptıktan sonra sonucu capraz_skor adlı değişkene atar.
print(capraz_skor) # bir önceki adımdaki capraz skor değişkenini ekrana yazdırdım sebebi de gelen sonuç yine modelin iyileştirilmeye gidilip gidilmeyeceğinde yol göstereceği için.
# ilerleyen aşamalarda kullanıcıdan bilgiler isteyip tahmin yaptıracağım fakat kullanıcıdan alacağım bilgilerde Binanın_Kat_Sayısı bilgisi, Banyo_Sayısı bilgisi olmayacak yani bu bilgileri kullanıcıdan almayacağım. Kullanıcıdan bu bilgileri almadığımda bu alanların verileri boş olur ve bu yüzden model tahmin yapamaz. Bu verilerde eğitim verisinin medyan değerleri kullanılarak atanmalıdır, böylece tahmin matrisi modelin beklediği formatta eksiksiz hale gelir. Bu işlemi gerçekleştirmek için şu satırları yazdım:
median_kat_sayisi = dr_coded['Binanın_Kat_Sayısı'].median()
median_banyo_sayisi = dr_coded['Banyo_Sayısı'].median()

# kullanıcıdan şehir, net metrekare ve oda sayısı verilerini girmesini istedim:
kullanici_sehir = input("Şehir: ")
kullanici_net_metrekare = float(input("Net Metrekare: "))
kullanici_oda = int(input("Oda Sayısı: "))
# Kullanıcıdan veriler alındığında modelin eğitildiği kolon başlıkları adı ve sırasıyla yapılmasını sağlamak için yapılan adımlar:
tahmin_sutunlari = X_train.columns.tolist() # x_train adlı değişkende eğitim verilerinin hedef değişken (fiyat) dışındaki özellikleri (Net metrekare, oda sayısı, ısıtma tipi vs.) yazılıydı. x_train.columns demek X_train tablosundaki tüm sütun başlıklarının listesini vermek demektir. tolist() komutu ile de bu sütun başlıklarını bir Python listesine çevirir. Yani tahmin_sutunları adlı değişkende x_train veri setindeki kolonların başlıkları bir python listesi şeklinde tutulması için bu satırı yazdım. 
kullanici_verisi_tablosu = pd.DataFrame(columns=tahmin_sutunlari) # pd.DataFrame boş tablo oluşturur bu boş tablonun kolonları şunlar olsun dedik-> tahmin_sutunlari değişkenindeki liste. tahmin_sutunları adlı liste de zaten x_train listesindeki tablonun tüm sütun başlıklarını yani kısaca hedef değişken dışındaki veri setinde bulunan diğer kolon başlıklarını içeriyordu.
kullanici_verisi_tablosu.loc[0] = 0 # Tablonun ilk satırındaki tüm hücrelere başlangıçta 0 ile doldurur.
kullanici_verisi_tablosu.loc[0, 'Net_Metrekare'] = kullanici_net_metrekare # kullanici_net_metrekare değişkeni kullanıcının girdiği net metrekare değerini tutar. Bu durumda bu satır kullanıcıdan gelen net metrekare değerini hazırlanan boş tabloda doğru hücreye yazılmasını sağlar. Bir önceki satırda ilk hücrelere 0 yazılmıştı bu satır ile net_metrekare kolonuna kullanıcının girdiği net metrekare verisi girilir yani 0 gider kullanıcının girdiği net metrekare verisi gelir. 
kullanici_verisi_tablosu.loc[0, 'Oda_Sayısı'] = kullanici_oda # bir önceki satırda yapılanların aynısı oda sayısı alanı için de yapılır (kullanıcının girdiği oda sayısı değeri kullanici_oda adlı değişkende tutuluyordu)
kullanici_verisi_tablosu.loc[0, 'Brüt_Metrekare'] = float(kullanici_net_metrekare * 1.25) # Brüt Metrekare: Kullanıcıdan alınmadığı için Net Metrekarenin %25 fazlasını varsayıyorum.
kullanici_verisi_tablosu.loc[0, 'Binanın_Kat_Sayısı'] = median_kat_sayisi # kullanıcıdan binanın katsayıyısı değeri alınmaz zaten bu durumu 67. satırda açıkladım. kısaca 81. satır gereği Tablonun ilk satırındaki tüm hücrelerini 0 ile doldurtmuştum Binanın_Kat_Sayısı alanının ilk hücresinde de doğal olarak başta 0 yazıyordu fakat bu satırdaki kod gereği median_kat_sayisi değişkeninden gelen değer atandı
kullanici_verisi_tablosu.loc[0, 'Banyo_Sayısı'] = median_banyo_sayisi # bir önceki satırda yapılanların aynısı aynı mantıkla Banyo_Sayısı alanı için de yapıldı.
# Kullanıcının girdiği şehir bilgisini (kullanıcı string veri tipinde şehir bilgisini girecek neticede...) One-Hot Encoding ile 0 veya 1'e çevirmek için şu satırları yazdım: (model sayılarla çalışır bu yüzden kullanıcının girdiği string veri tipindeki şehir bilgisi sayısallaştırılmalıydı bu yüzden de bloğu yazdım)
sehir_kolonu = f"Şehir_{kullanici_sehir}" 
if sehir_kolonu in tahmin_sutunlari:
    kullanici_verisi_tablosu.loc[0, sehir_kolonu] = 1
else:
    print(f"Uyarı: '{kullanici_sehir}' şehri eğitim verisinde bulunmadığı için tahmin kalitesi düşebilir.")
# Bu bloklar özetle şunu yapar-> başta tüm hücrelerde 0 yazıyordu sonra kullanıcı şehir girdi ve girdiği şehir modelin eğitim veri setinde varsa o hücre 0ken 1 oldu eğer kullanıcının girdiği şehir modelin eğitim veri setinde yoksa 92.satırdaki şekilde bir uyarı mesajı verdirtti.

# Kullanıcının gireceği metrekare verisi standartlaştırılmalı. Modelin eğitim ve test verilerinde tüm numeric olan alanları standartlaştırdım dolayısıyla model bu standartlaşmaya göre öğrendi. Kullanıcının gireceği metrekare verisi de standartlaştırılmalı ki modelin öğrendiği dilde olsun yani kullanıcı ile model aynı dilde konuşsun. Kullanıcının girdiği metrekare verisi standartlaştırılmazsa model daha önce görmediği bu kadar büyük ham sayılarla karşılaşır ve hata yapar. 
kullanici_verisi_tablosu[numeric_olanlar] = scaler.transform(kullanici_verisi_tablosu[numeric_olanlar]) 

x_user = np.array(kullanici_verisi_tablosu) # İnsanların okuması için kolay olan Pandas tablosunu, makinenin anlayacağı saf matematiksel sayı listelerine (matrise) (NumPy Array/Dizi) çevirmesi için bu satırı yazdım. Çünkü makine sadece sayılardan anlar; bu yüzden tablonun makinenin anlayacağı sayısal bir diziye/matrise çevrilmesi gerekiyordu.
tahmin = modell.predict(x_user.reshape(1, -1)) # predict komutu tahmin yapmayı sağlar ve yaptığı tahmini tahmin adlı değişkene atar. x_user.reshape(1,-1) ifadesi ise şunun için vardır: Kullanıcıdan biz oda sayısı, metrekare, şehir bilgilerini alıyoruz ya bu veriler tek boyutlu olarak saklanacak fakat modelin çalışması için matris şeklinde kullanıcıdan gelen bilgiler tutulmalı. İşte kullanıcıdan gelen tek boyutlu diziyi iki boyutlu dizi haline (matrise) getirmek için bu kısmı yazdım.
# gerekli bilgileri ekrana yazdırttım:
print(f"\nİstenen Ev Özellikleri:")
print(f"  > Şehir: {kullanici_sehir}")
print(f"  > Net Metrekare: {kullanici_net_metrekare}")
print(f"  > Oda Sayısı: {kullanici_oda}")
print(f" TAHMİNİ FİYAT: {tahmin[0]:,.2f} TL")

