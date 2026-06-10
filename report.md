---

title: "Miniaturowy model językowy do generowania krótkich historii"
author: "Mateusz Stawicki"
fontsize: 12pt
geometry: margin=2.5cm
linestretch: 1.5
----------------

# 1. Wprowadzenie

Dynamiczny rozwój dużych modeli językowych pokazał, że architektura Transformer może bardzo skutecznie modelować język naturalny. Jednocześnie klasyczne modele LLM są zwykle bardzo duże, kosztowne obliczeniowo i trudne do pełnego zrozumienia na poziomie implementacyjnym. Z tego powodu interesującym problemem jest sprawdzenie, czy można stworzyć mały model językowy, który mimo ograniczonej liczby parametrów będzie w stanie generować spójne, proste teksty.

Problemem realizowanym w projekcie było zbudowanie miniaturowego modelu językowego typu autoregresyjnego, którego zadaniem jest generowanie krótkich historii dla dzieci. Model nie miał być uniwersalnym chatbotem, lecz wyspecjalizowanym systemem przewidującym kolejne tokeny tekstu na podstawie wcześniejszego kontekstu. Takie zawężenie problemu pozwala trenować znacznie mniejszą sieć niż typowe współczesne modele językowe, a jednocześnie zachować sensowną jakość generowanego tekstu.

Celem projektu było zaimplementowanie kompletnego potoku pracy z małym modelem językowym: od przygotowania danych, przez tokenizację, definicję architektury Transformer, trening modelu, zapis wag, aż po uruchomienie inferencji z poziomu osobnego skryptu. Projekt miał również charakter edukacyjny: ważne było nie tylko uzyskanie działającego generatora tekstu, ale też zrozumienie najważniejszych elementów działania modelu językowego, takich jak tokenizacja BPE, mechanizm self-attention, maskowanie przyczynowe, funkcja straty cross-entropy i autoregresyjne generowanie tekstu.

# 2. Metody i dane

## 2.1. Opis metody

Zastosowana metoda opiera się na małym modelu Transformer typu decoder-only. Model działa autoregresyjnie, czyli w każdym kroku przewiduje następny token na podstawie tokenów już wygenerowanych. W trakcie treningu wejściem są fragmenty tekstu podzielone na sekwencje o stałej długości, a etykietami są te same sekwencje przesunięte o jeden token. Dzięki temu model uczy się klasycznego zadania language modeling: przewidywania kolejnego elementu tekstu.

Architektura modelu składa się z kilku głównych części. Pierwszą są embeddingi tokenów, które zamieniają identyfikatory tokenów na wektory liczbowe. Drugą są embeddingi pozycyjne, dzięki którym model otrzymuje informację o kolejności tokenów w sekwencji. Następnie dane przechodzą przez kilka bloków Transformer. Każdy blok zawiera mechanizm wielogłowicowej uwagi przyczynowej oraz prostą sieć feed-forward. W bloku zastosowano także normalizację warstwową oraz połączenia rezydualne, które ułatwiają uczenie modelu.

Mechanizm uwagi przyczynowej jest kluczowy dla poprawnego generowania tekstu. Model nie może „widzieć przyszłości”, dlatego w obliczeniach attention stosowana jest maska trójkątna. Blokuje ona dostęp do tokenów znajdujących się dalej w sekwencji niż aktualnie przewidywany token. Dzięki temu podczas treningu zachowana jest zgodność z późniejszą inferencją, w której tekst powstaje krok po kroku.

Tokenizacja została wykonana za pomocą algorytmu BPE. Zamiast operować bezpośrednio na znakach lub całych słowach, tekst jest dzielony na tokeny będące mniejszymi jednostkami. Jest to praktyczne, ponieważ pozwala obsługiwać słowa rzadkie, znaki interpunkcyjne oraz części wyrazów przy ograniczonym rozmiarze słownika. W projekcie przyjęto słownik o rozmiarze 4096 tokenów.

Model ma następujące główne hiperparametry: długość kontekstu 256 tokenów, wymiar embeddingu 256, 8 głów attention, 4 bloki Transformer oraz dropout równy 0.1 w trakcie treningu. Podczas inferencji dropout jest wyłączony. Liczba parametrów modelu wynosi około 4,26 mln, co klasyfikuje go jako bardzo mały model językowy w porównaniu ze współczesnymi modelami LLM.

## 2.2. Stan wiedzy

Podstawą współczesnych modeli językowych jest architektura Transformer zaproponowana w pracy *Attention Is All You Need*. Jej najważniejszą ideą jest zastąpienie rekurencji mechanizmem self-attention, który pozwala modelowi analizować zależności między tokenami w sekwencji. Dzięki temu Transformer dobrze skaluje się obliczeniowo i może być trenowany równolegle na GPU.

W kontekście małych modeli językowych szczególnie istotny jest zbiór TinyStories. Został on zaprojektowany do badania, jak małe mogą być modele językowe, aby nadal generować spójny tekst. Dane składają się z prostych, syntetycznie wygenerowanych historii, których słownictwo jest celowo ograniczone i zbliżone do języka zrozumiałego dla kilkuletnich dzieci. Taki zbiór danych jest bardzo dobry do eksperymentów edukacyjnych, ponieważ pozwala trenować małe modele bez konieczności używania ogromnych korpusów tekstu.

Projekt wpisuje się więc w podejście Small Language Models. Zamiast budować możliwie największy model, skupia się na ograniczonej domenie, prostych danych i architekturze możliwej do samodzielnej analizy. Pozwala to lepiej zrozumieć, które elementy są konieczne do działania modelu językowego, a które są związane głównie ze skalowaniem.

## 2.3. Technologie

Projekt został zaimplementowany w języku Python z użyciem biblioteki PyTorch. PyTorch odpowiada za definicję warstw sieci neuronowej, automatyczne różniczkowanie, trening na CPU lub GPU oraz zapis i odczyt wag modelu. Model został zdefiniowany jako klasa dziedzicząca po `nn.Module`, a jego elementy, takie jak attention, MLP i bloki Transformer, zostały zaimplementowane jako osobne klasy.

Do obsługi danych wykorzystano bibliotekę Hugging Face Datasets, która umożliwia pobranie zbioru TinyStories przez funkcję `load_dataset`. Tokenizacja została wykonana za pomocą biblioteki Hugging Face Tokenizers, a wytrenowany tokenizer zapisano do pliku `tinystories_bpe.json`. Sam model po treningu został zapisany do pliku `tinystories_model.pt`.

Inferencja została wydzielona do pliku `inference.py`. Skrypt ładuje tokenizer, tworzy model o tej samej architekturze co w treningu, wczytuje zapisane wagi, koduje prompt użytkownika i generuje tekst. Użytkownik może sterować generowaniem przez parametry takie jak liczba nowych tokenów, temperatura oraz `top_k`.

## 2.4. Opis danych

Dane pochodzą ze zbioru TinyStories. Jest to zbiór krótkich historii przeznaczonych do trenowania i ewaluacji małych modeli językowych. Każdy przykład jest tekstem w języku angielskim. Historie są proste, krótkie i mają ograniczone słownictwo, co pasuje do celu projektu: nauczenia małego modelu generowania podstawowych, spójnych opowiadań.

W projekcie dane zostały najpierw pobrane, a następnie użyte do wytrenowania tokenizera BPE. Następnie cały zbiór został przekształcony na sekwencje identyfikatorów tokenów. Tokeny zostały połączone i podzielone na bloki o długości 256, odpowiadające maksymalnej długości kontekstu modelu. Dla każdego bloku wejście i etykieta mają tę samą zawartość, natomiast funkcja straty wewnątrz modelu porównuje predykcje z tokenami przesuniętymi o jedną pozycję.

Do trenowania użyto mini-batchy o rozmiarze 64. Dane treningowe były tasowane, natomiast dane walidacyjne nie wymagały tasowania. Model był trenowany przez 2 epoki, z możliwością uruchomienia dodatkowej epoki po wczytaniu zapisanych wag.

# 3. Wyniki

Efektem projektu jest działający, kompletny system do generowania krótkich historii. System obejmuje notatnik treningowy oraz osobny skrypt inferencyjny. W notatniku zaimplementowano pełny proces: konfigurację hiperparametrów, definicję architektury Transformer, pobranie danych, trening tokenizera, tokenizację zbioru, przygotowanie DataLoaderów, trening modelu, walidację oraz zapis wag.

Model po uruchomieniu wykonuje klasyczny przepływ inferencji. Najpierw prompt użytkownika jest zamieniany na tokeny. Następnie model przewiduje rozkład prawdopodobieństwa kolejnego tokenu. Rozkład ten jest modyfikowany przez temperaturę i opcjonalne ograniczenie `top_k`, a potem losowany jest następny token. Token zostaje dopisany do sekwencji i cały proces powtarza się aż do osiągnięcia zadanej liczby tokenów albo wygenerowania tokenu końca tekstu.

Wynikiem praktycznym jest możliwość generowania prostych historii w stylu danych TinyStories. Model nie ma wiedzy ogólnej porównywalnej z dużymi LLM, ale potrafi tworzyć tekst zgodny z wąską domeną treningową. Wygenerowane historie powinny mieć prostą składnię, nieskomplikowane słownictwo i strukturę przypominającą krótkie opowiadania dziecięce.

Istotnym rezultatem jest również poprawne rozdzielenie etapu treningu od etapu inferencji. Dzięki zapisaniu tokenizera i wag modelu możliwe jest uruchomienie generatora bez ponownego trenowania sieci. Jest to ważne, ponieważ odpowiada typowemu sposobowi pracy z modelami uczenia maszynowego: trening jest kosztowny i wykonywany rzadziej, natomiast inferencja powinna być możliwie prosta do uruchomienia.

Podczas eksperymentów można zaobserwować wpływ parametrów generowania. Niższa temperatura powoduje bardziej zachowawcze i przewidywalne teksty, natomiast wyższa temperatura zwiększa losowość. Parametr `top_k` ogranicza wybór następnego tokenu do najbardziej prawdopodobnych kandydatów, co zmniejsza ryzyko generowania bardzo przypadkowych słów. Dla małego modelu takie ograniczenia są szczególnie przydatne, ponieważ stabilizują generowany tekst.

Najważniejszą obserwacją jest to, że jakość modelu wynika nie tylko z architektury, ale również z dopasowania skali problemu do skali modelu. Model o kilku milionach parametrów prawdopodobnie nie poradziłby sobie dobrze jako ogólny model językowy, ale w ograniczonej domenie prostych historii może dawać sensowne wyniki. Pokazuje to, że małe modele językowe mogą być użyteczne, jeśli zadanie jest dobrze dobrane.

# 4. Wnioski

Projekt pokazuje, że możliwe jest samodzielne zbudowanie małego modelu językowego od poziomu danych i tokenizacji aż po działającą inferencję. Najważniejszym elementem systemu jest decoder-only Transformer z maskowaną self-attention, ponieważ umożliwia przewidywanie kolejnych tokenów bez dostępu do przyszłego kontekstu.

Ważnym wnioskiem jest to, że małe modele językowe wymagają dobrze ograniczonego zadania. TinyStories jest dobrym wyborem, ponieważ zawiera proste teksty o ograniczonym słownictwie. Dzięki temu nawet niewielki model może nauczyć się podstawowych wzorców składniowych i narracyjnych.

Projekt ma także wartość edukacyjną. Implementacja pozwala zrozumieć praktyczne znaczenie takich pojęć jak tokenizacja BPE, embeddingi pozycyjne, multi-head attention, połączenia rezydualne, cross-entropy loss, dropout, AdamW oraz autoregresyjne generowanie. W porównaniu z użyciem gotowego modelu z biblioteki, taka implementacja lepiej pokazuje, co dzieje się „pod spodem”.

Ograniczeniem projektu jest niewielka skala modelu i brak rozbudowanej ewaluacji jakościowej. W przyszłości można rozszerzyć projekt o automatyczne porównywanie różnych konfiguracji, zapis historii strat treningowych, pomiar perplexity, generowanie wielu próbek dla tych samych promptów oraz ocenę wpływu liczby warstw, długości kontekstu i rozmiaru słownika na jakość tekstu.

# 5. Bibliografia

1. Vaswani, A. et al. (2017). *Attention Is All You Need*. arXiv:1706.03762.
2. Eldan, R., Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* arXiv:2305.07759.
3. Hugging Face. *roneneldan/TinyStories dataset*.
4. PyTorch Documentation. *torch.nn, torch.optim.AdamW, torch.utils.data.DataLoader*.
5. Hugging Face Documentation. *Tokenizers and Byte-Pair Encoding*.
6. Repozytorium projektu: `Makon324/mini_LLM`, GitHub.
