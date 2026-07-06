# Data Dictionary - White Wine Quality Dataset

| Nama Fitur | Tipe Data | Deskripsi |
|---|---|---|
| fixed acidity | Float | Asam utama pada anggur (tartarat) yang tidak mudah menguap. |
| volatile acidity | Float | Asam asetat pada anggur yang jika terlalu tinggi merusak rasa (rasa cuka). |
| citric acid | Float | Asam sitrat, memberikan kesegaran pada wine. |
| residual sugar | Float | Sisa gula setelah proses fermentasi selesai. |
| chlorides | Float | Jumlah garam (klorida) di dalam cairan wine. |
| free sulfur dioxide | Float | Bentuk gas $SO_2$ bebas, mencegah pertumbuhan mikroba & oksidasi. |
| total sulfur dioxide | Float | Total keseluruhan $SO_2$ bebas dan terikat. |
| density | Float | Kerapatan massa cairan wine (mendekati air tergantung gula/alkohol). |
| pH | Float | Tingkat keasaman cairan (skala 0-14, white wine biasanya 3.0-3.5). |
| sulphates | Float | Aditif kalium sulfat yang berkontribusi pada gas sulfur dioksida. |
| alcohol | Float | Persentase kadar alkohol (%) dalam wine. |
| quality | Integer | Skor target sensorik asli (0-10). |
| target | Integer | Label biner hasil transformasi (1: Premium [skor >= 6], 0: Normal [skor < 6]). |