# Laporan Proyek Machine Learning - Roni Merdiansah

## Project Overview

Beralihnya tren game dari *game console* menjadi *digital game* merubah cara pendistribusian game tersebut. Game digital diakui memiliki banyak keunggulan
dibandingkan game console, diantaranya; mudah diakses, bersifat *online*, dll. Pendistribusian game digital dilakukan melalui platform-platform tertentu. 
Banyaknya genre game yang ada juga membuat para pemain bingung dalam memilih game apa yang akan mereka mainkan. Salah satu cara untuk mengatasi masalah tersebut 
adalah menggunakan Sistem Rekomendasi pada setiap platform game.

Sistem rekomendasi adalah aplikasi perangkat lunak yang bertujuan untuk mendukung pengguna dalam pengambilan keputusan saat berinteraksi dengan ruang informasi
yang besar. Sistem rekomendasi membantu mengatasi masalah informasi yang berlebihan dengan mengekspos pengguna ke item yang paling menarik, terbaru, dan relevan 
[[1](https://www.hindawi.com/journals/sp/2022/6084363/)]. Sistem ini akan merekomendasikan game berdasarkan genre yang pemain sukai, ataupun
berdasarkan game apa yang sedang tren saat itu.

## Business Understanding

### Problem Statements

Bagaimana cara merekomendasikan game yang dimainkan pemain lain dapat direkomendasikan kepada pemain lainnya juga?

### Goals

Dapat membuat sistem rekomendasi yang akurat berdasarkan ratings dan aktivitas pemain pada masa lalu.

### Solution statements

Proyek ini menggunakan 2 algoritma Machine Learning sistem rekomendasi sebagai solusi, diantaranya :

* *Content Based Filtering*: adalah algoritma yang merekomendasikan game serupa dengan genre yang disukai pemain, berdasarkan tindakan mereka sebelumnya atau umpan balik eksplisit.
* *Collaborative Filtering*: adalah algoritma yang bergantung pada pendapat komunitas pengguna, tidak memerlukan atribut untuk setiap itemnya.

Algoritma Content Based Filtering digunakan untuk merekemondesikan game berdasarkan aktivitas pemain pada masa lalu, sedangkan algoritma Collabarative Filltering digunakan untuk merekomendasikan game berdasarkan rating yang paling tinggi.

## Data Understanding

Dataset ini berisi data penjualan video game dari seluruh dunia, di berbagai platform, genre, dan wilayah. Dataset ini memberikan informasi tentang apa yang dianggap sebagai game hit di industri game saat ini [[Discovering Hidden Trends in Global Video Games](https://www.kaggle.com/datasets/thedevastator/discovering-hidden-trends-in-global-video-games)]. Meskipun dataset ini merupakan data penjualan game, pada proyek ini akan mencoba menggunakan data yang ada untuk membuat sistem rekomendasi. Adapun total data ini berjumlah 1907.

### Variable

Variabel-variabel pada Discovering Hidden Trends in Global Video Games dataset adalah sebagai berikut:
- `Rank`	Peringkat game dalam hal penjualan global. (Int)
- `Game Title` : Judul game. (String)
- `Platform` : Platform tempat game dirilis. (String)
- `Year` : Tahun game dirilis. (Integer)
- `Genre` : Genre game. (String)
- `Publisher` : Penerbit game. (String)
- `North America` : Penjualan game di Amerika Utara. (Integer)
- `Europe` : Penjualan game di Eropa. (Integer)
- `Japan` : Penjualan game di Jepang. (Integer)
- `Rest of World` : Penjualan game di seluruh dunia. (Integer)
- `Global` : Total penjualan game secara global. (Integer)
- `Review` : Skor ulasan game. (Float)

### Exploratory Data Analysis

![](https://github.com/Dapperson/Machine-Learning-Terapan/blob/main/Proyek%20Akhir/Image/Platform.png)

$$Gambar\ 1.1\ Platform$$

![](https://github.com/Dapperson/Machine-Learning-Terapan/blob/main/Proyek%20Akhir/Image/Publisher.png)

$$Gambar\ 1.2\ Publisher$$

![](https://github.com/Dapperson/Machine-Learning-Terapan/blob/main/Proyek%20Akhir/Image/Genre.png)

$$Gambar\ 1.3\ Genre$$

![](https://github.com/Dapperson/Machine-Learning-Terapan/blob/main/Proyek%20Akhir/Image/Tahun%20Rilis.png)

$$Gambar\ 1.4\ Year$$

Dari visualisasi diatas didapatkan beberapa *insight* diantaranya
- Terlihat pada $Gambar\ 1.1$ bahwa Platform yang paling sering merilis game adalah PS2
- Perusahaan penerbit game berdasarkan $Gambar\ 1.2$, paling banyak adalah Nintendo
- Game dengan genre **Sport** paling banyak dibuat, dilihat dari $Gambar\ 1.3$
- Jumlah game dirilis tertinggi pada tahun 2008 berdasarkan $Gambar\ 1.4$

## Data Preparation
Terdapat beberapa tahapan yang dilakukan sebelum membuat model sistem rekomendasi. Adapun tahapan nya sebagai berikut:

1. **Membuat Fitur `User_ID`** : Data ini tidak memiliki fitur user_id, maka diasumsikan bahwa setiap baris yang ada pada kolom adalah user_id, sehingga setiap index pada baris di labeli sebagai user_id.
2. **Memilih Fitur** : Data ini merupakan data penjualan dan ini adalah proyek membuat sistem rekomendasi, maka dilakukan pemilihan fitur terlebih dahulu guna mempermudah dalam membuat model. Fitur yand dipilih sebagai berikut:
 - `User_ID` 
 - `Game Title`	
 - `Platform`	
 - `Year`	
 - `Genre`	
 - `Publisher` 
 - `Review`
3. **Format Ulang Fitur** : Melakukan pembulatan bilangan pada kolom `Review`
4. **Membuat Fitur `Game_ID`** : Sama seperti sebelumnya, dataset ini tidak memiliki fitur game_id sehingga diasumsikan bahwa setiap nilai *Unique* pada `Game Title` adalah game_id kemudian melabelinya
5. **Menghapus Duplikasi Data** : Menghilangkan data duplikasi berdasarkan fitur `Game_ID`
6. **Membuat Dictionary** : Dilakukan untuk membuat DataFrame baru dengan beberapa fitur saja
7. **Membuat DataFrame** : DataFrame yang dibuat hanya berisikan beberapa fitur saja untuk keperluan pembuatan model
8. **Mengaplikasikan TF-IDF Vectorizer** : Menghitung bobot setiap kata terhadap data yang ada
9. **Menghitung Cosine Similarity** : Bertujuan untuk mengetahui nilai kemiripan antar data

## Modeling dan Results

### A. Model Development dengan *Content Based Filtering*

Setelah mengaplikasikan *TF-IDF Vectorizer* dan melihat *Cosine Similarity* selanjutnya bisa melakukan ke tahap berikutnya yaitu membuat model *Content Based Filtering*. TF-IDF Vectorizer digunakan pada sistem rekomendasi untuk menemukan representasi fitur penting dari setiap genre game. Sedangkan *Cosine Similarity* berfungai untuk menghitung derajat kesamaan (similarity degree) antar genre dengan teknik cosine similarity. Konsep dari model ini adalah merekomendasikan daftar game berdasarkan index kemiripan dari game yang pernah dimainkan sebelumnya. Untuk lebih jelasnya bisa dilihat dibawah ini

$Tabel\ 1.1\ Game\ yang\ pernah\ di\ mainkan$
| game_id | genre  |                         title |
| ------- | -------|------------------------------ |
|     404 | Action	| Grand Theft Auto: San Andreas |

Berdasarkan $Tabel\ 1.1$ bahwa game yang pernah dimainkan pemain adalah **Grand Theft Auto: San Andreas** dengan genre **Action**. Dari data tabel tersebut sistem dapat merekomendasikan 5 game teratas dengan index kesamaan yang tinggi. Adapun kelima game tersebut sebagai berikut:

$Tabel\ 1.2\ Hasil\ rekomendasi\ model$
|                         title	|  genre |
| ----------------------------- | ------ |
| New Super Mario Bros. U	      | Action |
| Final Fight 2	                | Action |
| The Lost World: Jurassic Park	| Action |
| Frogger 2: Swampy's Revenge	  | Action |
| Dynasty Warriors 4	           | Action |

Dari $Tabel\ 1.2$ dapat dilihat bahwa model dapat bekerja dengan baik dengan merekomendasikan game yang memiliki genre **Action** dikarenakan pemain sebelumnya memainkan game **Grand Theft Auto: San Andreas** yang memiliki genre **Action**. Model menganggap bahwa pemain akan menyukai game dengan genre serupa namun dengan judul game yang berbeda.

### B. Model Development dengan *Collaborative Filtering*

Model ini memilki konsep merekomendasikan game berdasarkan rating yang diberikan pemain lain. Sebelum membuat model ini, data harus melalui proses *Training* terlebih dahulu. Sebelum malakukan training, data harus di bagi terlebih dahulu menjadi data *training* dan data *validation*, dengan skala perbandingan data adalah 80:20, dan menggunakan *activation sigmoid*. Namun sebelumnya, perlu memetakan (mapping) data user dan game menjadi satu value terlebih dahulu, kemudian membuat rating dalam skala 0 sampai 1 agar mudah dalam melakukan proses training. Setelah melakukan semua proses itu maka selanjutnya melakukan uji coba pada model

$Tabel\ 2.1\ Hasil\ rekomendasi\ untuk\ user$
| Showing recommendations for users: 676|         |
| ------------------------------------- | ------- |
| Game with high ratings from user      |         |
| ------------------------------------- | ------- |
| Devil May Cry 2                       |  Action |
| ------------------------------------- | ------- |
| Top 10 Game Recommendation            |         |
| ------------------------------------- | ------- |
| Call of Duty: Modern Warfare 2        | Shooter |
| GoldenEye 007                         | Shooter |
| Uncharted 2: Among Thieves            |  Action |
| Metal Gear Solid 2: Sons of Liberty   |  Action |
| God of War                            |  Action |
| Metroid Prime                         | Shooter | 
| BioShock                              | Shooter |
| Perfect Dark                          |  Action |
| NFL 2K1                               |  Sports |
| Burnout Revenge                       |  Racing |

Dari tabel $Tabel\ 2.1$ bisa dilihat bahwa model dapat merekomendasikan game untuk `user_id` : **676** berdasarkan rating tinggi dari pemain lain adalah game dengan genre **Action**. Sedangkan untuk Top 10 rekomendasi game didominasi memiliki genre **Action** dan **Shooter**.  

## Evaluation
### A. *Content Based Filtering*

Untuk evaluasi pada model ini sederhana saja dengan menggunakan rumus *precision* sebagai berikut

![](https://github.com/Dapperson/Machine-Learning-Terapan/blob/main/Proyek%20Akhir/Image/Precision.jpeg)

$$Gambar\ 2.1\ Rumus\ Precision$$

Berdasarkan $Gambar\ 2.1$ yang merupakan rumus dari precision, maka cara menghitung nilai precision berdasarkan model yang telah diuji cobakan adalah **P = 5/5** atau nilai precision nya adalah **100%**. Nilai 5 didapatkan dari genre yang muncul pada rekomendasi semuanya adalah **Action**, relavan dengan genre game yang pernah dimainkan sebelumnya oleh pemain.

### B. *Collaborative Filtering*

Model ini menggunakan *Binary Crossentropy* untuk menghitung loss function, *Adam (Adaptive Moment Estimation)* sebagai optimizer, dan *Root Mean Squared Error (RMSE)* sebagai metrics evaluation. 

![](https://github.com/Dapperson/Machine-Learning-Terapan/blob/main/Proyek%20Akhir/Image/RMSE.png)

$Gambar\ 3.1\ Rumus\ RMSE\$

Berikut penjelasan dari $Gambar\ 3.1$
- At = Nilai data Aktual
- Ft = Nilai hasil prediksi
- N = banyaknya data
- ∑ = Summation (Jumlahkan keseluruhan  nilai) [[2](https://www.khoiri.com/2020/12/cara-menghitung-root-mean-square-error-rmse.html)]

RMSE adalah metode pengukuran dengan mengukur perbedaan nilai dari prediksi sebuah model sebagai estimasi atas nilai yang diobservasi. Root Mean Square Error adalah hasil dari akar kuadrat Mean Square Error. Keakuratan metode estimasi kesalahan pengukuran ditandai dengan adanya nilai RMSE yang kecil. Metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih kecil dikatakan lebih akurat daripada metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih besar. [[3](https://www.dqlab.id/kriteria-jenis-teknik-analisis-data-dalam-forecasting)]

Kelebihan dari metriks ini adalah menghukum kesalahan besar lebih sehingga bisa lebih tepat dalam beberapa kasus. Sedangkan kekurangan dari metriks ini adalah memberikan bobot yang relatif tinggi untuk kesalahan besar. Ini berarti RMSE harus lebih berguna ketika kesalahan besar sangat tidak diinginkan.

Berikut hasil visualisasi nya

![](https://github.com/Dapperson/Machine-Learning-Terapan/blob/main/Proyek%20Akhir/Image/RSME%20Visualisasi.png)

$Gambar\ 3.2\ Visualisasi\ proses\ training$

Visualisasi model training pada $Gambar\ 3.2$ menggunakan konvergen pada epochs sebanyak 100, dan didapatkan nilai akhir sebagai berikut
- **loss** : 0.5190
- **root_mean_squared_error** : 0.0375
- **val_loss** : 0.6936
- **val_root_mean_squared_error** : 0.2793

Perhatikan bahwa garis **test** lurus horizontal yang artinya memiliki nilai yang konstan. Hal ini terjadi karena dataset merupakan satu kesatuan dan juga antara variabel `user` dengan variabel pembandingnya, yaitu `game` tidak memiliki fitur `id` masing-masing. Fitur `id` didapatkan melalui proses pelabelan sehigga antara `user` dan `game` memiliki jumlah nilai *Unique* yang sama.

## Conclusion
Setelah membuat sistem rekomendasi menggunakan dua buah model yaitu *Content Based Filtering* dan *Collaborative Filtering* dapat dilihat bahwa hanya satu model yang memiliki kemampuan untuk merekomendasikan dengan akurasi tinggi, yaitu *Content Based Filtering*. Sedangkan untuk model *Collaborative Filtering* terbilang masih kurang efektif, dikarenakan selisih antara nilai **root_mean_squared_error** dengan **val_root_mean_squared_error** cukup besar yaitu 0.2418, sehingga masih memerlukan improvisasi lagi. Pemilihan dataset yang tepat juga dapat menjadi solusi untuk membuat model yang memiliki akurasi yang tinggi. 

Dengan kata lain sistem rekomendasi yang telah dibuat dalam proyek ini hanya bisa merekomendasikan game berdasarkan aktivitas pemain dimasa lalu, dan masih kurang memenuhi standar dalam memberikan rekomendasi game berdasarkan rating yang diberikan oleh pemain lain.

## Referensi
[1] [Fasiha Ikram, "Multimedia Recommendation System for Video Game Based on High-Level Visual Semantic Features"](https://www.hindawi.com/journals/sp/2022/6084363/)

[2] [Khoiri, "Pengertian dan Cara Menghitung Root Mean Square Error (RMSE)"](https://www.khoiri.com/2020/12/cara-menghitung-root-mean-square-error-rmse.html)

[3] [Reyvan Maulid, "Kriteria Jenis Teknik Analisis Data dalam Forecasting"](https://www.dqlab.id/kriteria-jenis-teknik-analisis-data-dalam-forecasting)
