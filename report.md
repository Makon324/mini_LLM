## 1. Wstęp
Celem projektu było zaprojektowanie, zaimplementowanie od podstaw w środowisku PyTorch oraz wytrenowanie autorskiego, małego modelu językowego bazującego na architekturze Transformer. Model został stworzony z myślą o generowaniu krótkich opowiadań na podstawie zbioru danych roneneldan/TinyStories.

Kluczowym założeniem projektu nie była wyłącznie optymalizacja metryk technicznych czy minimalizacja błędu podczas treningu, lecz stworzenie modelu zdolnego do generowania historii o rzeczywistej wartości dla docelowych odbiorców - dzieci. Jednocześnie zadbano o to, by model był na tyle mały, aby mógł generować opowiadania natychmiastowo, bezpośrednio na lokalnym urządzeniu użytkownika. W tym celu zaimplementowano również skrypt inferencyjny, który pozwala na dostosowanie parametrów generacji (np. temperatury czy top-k) oraz zdefiniowanie początku historii za pomocą promptu.

Realizacja tych założeń wymagała ręcznego zaprogramowania wszystkich elementów sieci (takich jak mechanizm Causal Self-Attention, osadzenia pozycyjne czy warstwy Feed-Forward) oraz wytrenowania własnego tokenizatora BPE.

## 2. Metoda

### 2.1 Dane
Wykorzystano zbiór danych TinyStories z paperu TinyStories (Eldan i Li, 2023, będący publicznie dostępnym korpusem syntetycznych historii wygenerowanych przez modele GPT-3.5 oraz GPT-4. Opowiadania zostały wygenerowane przy użyciu słownictwa typowego dla 3- lub 4-letniego dziecka w języku angielskim. Historie są bardzo krótkie, spójne gramatycznie i zawierają prostą narrację (często morał lub podstawowe interakcje między postaciami).

### 2.2 State of the art

#### Wprowadzenie do Małych Modeli Językowych (SLM)
W ostatnich latach rozwój modeli językowych (LLM) zdominowany był przez skalowanie parametrów. Jednak rosnące koszty obliczeniowe i bariery wejścia zwróciły uwagę badaczy na Małe Modele Językowe (Small Language Models - SLM). Obecny stan wiedzy wskazuje, że jakość danych treningowych jest równie istotna, co rozmiar modelu.

#### Oryginalna implementacja TinyStories
Kluczowym punktem odniesienia w tej dziedzinie jest praca TinyStories (Eldan i Li, 2023), która udowodniła, że modele rzędu kilku milionów parametrów potrafią generować spójny tekst. Jednakże, oryginalne modele TinyStories opierają się na suboptymalnych decyzjach architektonicznych – korzystają ze standardowego tokenizera GPT-2 o rozmiarze słownika wynoszącym 50 257 tokenów. W przypadku najmniejszych modeli skutkuje to drastycznym ograniczeniem liczby parametrów, które można przeznaczyć na mechanizmy uwagi (attention) i warstwy sprzężenia w przód (feed-forward), odpowiedzialne za faktyczne wnioskowanie i poprawność gramatyczną.

### 2.3 Technologia

Do realizacji projektu wykorzystano ekosystem języka Python oraz dedykowane biblioteki uczenia maszynowego:

* **PyTorch:** Główny framework obliczeniowy. Został użyty do budowy niestandardowej architektury modelu od podstaw, obsługi procesu propagacji wstecznej (backpropagation) oraz akceleracji sprzętowej na kartach graficznych (CUDA).
* **Hugging Face Tokenizers:** Zastosowano do wygenerowania autorskiego tokenizatora opartego na kodowaniu BPE (Byte-Pair Encoding), zoptymalizowanego pod mały słownik.
* **Hugging Face Datasets:** Biblioteka użyta do pobrania, strumieniowego przetwarzania korpusu tekstowego w paczkach (batching) oraz równoległej tokenizacji danych (multiprocessing).
* **Jupyter Notebook:** Środowisko wykorzystane do iteracyjnego eksperymentowania z architekturą, strojenia hiperparametrów oraz uruchamiania pętli treningowych.

### 2.4 Tokenizacja i Przygotowanie Danych

Na podstawie pobranych danych wygenerowano własny słownik BPE o rozmiarze 4096 tokenów.

Proces przygotowania tekstu obejmował:

* **Pre-tokenizację:** Wykorzystano podział tekstu na słowa bazujący na białych znakach (Whitespace).
* **Tokeny specjalne:** Wprowadzono tokeny systemowe: `<unk>` (nieznane znaki), `<bos>` i `<eos>` (odpowiednio początek i koniec historii) oraz `<pad>` (wypełnienie okna kontekstu).
* **Post-processing:** Każda historia podczas tokenizacji była automatycznie otaczana tokenami początku i końca sekwencji.
* **Grupowanie kontekstu:** Zamiast trenować na pojedynczych, różnej długości zdaniach, wszystkie tokeny połączono w jeden ciąg, a następnie podzielono na stałe, nienakładające się okna kontekstowe o maksymalnej długości 256 tokenów.

### 2.5 Architektura Modelu

Zaprojektowana sieć neuronowa składa z następujących elementów strukturalnych:

* **Warstwa Osadzeń (Embeddings):** Wektoryzacja tokenów (Token Embeddings) oraz ich pozycji (Positional Embeddings). Rozmiar przestrzeni osadzeń wynosi 256 wymiarów.
* **Współdzielenie Wag (Weight Tying):** Zastosowano kluczową optymalizację polegającą na połączeniu wag macierzy osadzeń tokenów z liniową warstwą wyjściową (LM Head). Pozwoliło to na znaczną redukcję ogólnej liczby parametrów modelu.
* **Bloki Transformera (Transformer Blocks):** Model składa się z 4 warstw transformera. Każdy blok zawiera:
* **Mechanizm Uwagi (Causal Self-Attention):** 8 głowic (Attention Heads), co daje 32 wymiary na głowicę. Zastosowano maskowanie przyczynowe (causal mask), uniemożliwiające modelowi "zaglądanie w przyszłość" poprzez wyzerowanie wag powyżej głównej przekątnej.
* **Wielowarstwowy Perceptron (MLP):** Sieć typu feed-forward rozszerzająca wymiarowość 4-krotnie, po której następuje nieliniowa funkcja aktywacji **GELU**, powracająca następnie do bazowych 256 wymiarów.
* **Normalizacja i Połączenia Resztkowe:** Użyto warstw `LayerNorm` (aplikowanych przed atencją oraz przed MLP) oraz połączeń typu skip-connection (residual), stabilizujących przepływ gradientu.



Zastosowano standardowy wzór na  atencję:


$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} \cdot M\right)V$$


### 2.6 Proces Uczenia

Model był trenowany na procesorze graficznym z wykorzystaniem funkcji straty Cross-Entropy Loss. Szczegóły procesu:

* **Optymalizator:** AdamW z współczynnikiem uczenia (learning rate) równym 5e-4 oraz regularyzacją weight decay na poziomie 0.01.
* **Regularyzacja:** Zastosowano warstwy Dropout o wartości 0.1 w mechanizmach uwagi oraz MLP, co miało na celu redukcję zjawiska przeuczenia (overfitting). Ponadto zastosowano obcinanie gradientów (gradient clipping) do maksymalnej wartości normy równej 1.0.
* **Epoki i Batching:** Przetworzone bloki danych zasilały pętlę uczącą w pakietach wielkości 64 próbek (batch size). Trening odbył się w cyklu obejmującym łącznie 3 epoki na całym zbiorze danych.

### 2.7 Inferencja i Generowanie Tekstu

W celu ewaluacji oraz praktycznego wykorzystania modelu zaimplementowano skrypt do generowania tekstu autoregresyjnie, znak po znaku. Podczas inferencji wyłączono mechanizmy regularyzacyjne (Dropout = 0.0).

Proces generowania kontrolowany jest przez dwie techniki próbkowania (sampling):

* **Temperatura (Temperature):** Skalowanie wartości wyjściowych (logitów) przed nałożeniem funkcji softmax (domyślnie ustawione na 0.8), co pozwala na płynną regulację między deterministycznym, a bardziej zróżnicowanym językiem.
* **Top-K Sampling:** Obcięcie rozkładu prawdopodobieństwa wyłącznie do *K* (domyślnie 10) najbardziej prawdopodobnych następnych tokenów.

#### Post-processing tekstu i korekta interpunkcji

Ze względu na specyfikę działania tokenizera BPE oraz zastosowanie pre-tokenizacji dzielącej tekst według białych znaków (Whitespace), proces dekodowania (zamiany wygenerowanych identyfikatorów z powrotem na tekst jawny) domyślnie łączy tokeny sekwencyjnie, wstawiając między nimi spacje. Skutkuje to powstawaniem błędów typograficznych w wynikowym ciągłym tekście, przejawiających się obecnością sztucznych, nadmiarowych odstępów bezpośrednio przed znakami interpunkcyjnymi (np. `"tekst ,"` zamiast `"tekst,"`).

Aby wygenerowane opowiadania były w pełni naturalne w odbiorze i poprawne gramatycznie, zaimplementowano dedykowany etap post-processingu (czyszczenia tekstu). Po przetłumaczeniu wektora tokenów na postać tekstową, skrypt wykonuje operacje wyszukiwania i zamiany, które automatycznie usuwają niepożądane spacje wokół kluczowych znaków interpunkcyjnych, w tym:

* przecinków (`" ,"` $\rightarrow$ `","`),
* kropek (`" ."` $\rightarrow$ `"."`),
* wykrzykników i pytajników (`" !"` $\rightarrow$ `"!"`, `" ?"` $\rightarrow$ `"?"`),
* dwukropków i średników (`" :"` $\rightarrow$ `":"`, `" ;"` $\rightarrow$ `";"`),
* cudzysłowów oraz apostrofów.

Warto zaznaczyć, że przyjęta metoda tokenizacji i łączenia znaków skutkuje niekiedy pojawieniem się nadmiarowej spacji wewnątrz pojedynczego słowa (zwłaszcza w przypadku słów nietypowych, rozbitych na mniejsze sub-tokeny BPE). W obecnej implementacji nie wprowadzono dodatkowych mechanizmów korygujących ten specyficzny artefakt, jednakże jest to rzadkie i nie stanowi znaczącego problemu.

Generowanie historii kończy się automatycznie po napotkaniu przez model specjalnego tokenu końca sekwencji `<eos>` lub po osiągnięciu odgórnie założonego limitu nowo utworzonych tokenów.

## 3. Wyniki

### 3.1. Prezentacja działającego systemu i instrukcja obsługi skryptu (`inference.py`)

W wyniku realizacji projektu uzyskano działający mały model językowy. Model działa autoregresyjnie: otrzymuje początkowy prompt, przewiduje następny token, dołącza go do sekwencji i powtarza ten proces.

System uruchamia się z poziomu wiersza poleceń za pomocą dedykowanego skryptu inferencyjnego. Skrypt automatycznie sprawdza dostępność akceleracji sprzętowej GPU (`cuda`) i w razie jej braku domyślnie przełącza obliczenia na procesor (`cpu`). W celach bezpieczeństwa wagi modelu ładowane są z flagą `weights_only=True`. Wywołanie metody `model.eval()` gwarantuje, że warstwy regularyzacyjne Dropout zostaną wyłączone, co stabilizuje proces deterministycznego i stochastycznego wnioskowania.

Po zakończeniu generowania tekstu skrypt dokonuje automatycznego oczyszczania i formatowania końcowego poprzez usuwanie zbędnych spacji przed podstawowymi znakami interpunkcyjnymi (takimi jak przecinki, kropki, wykrzykniki, pytajniki, dwukropki, średniki oraz cudzysłowy).

#### Wymagania wstępne

Przed uruchomieniem skryptu należy upewnić się, że w tym samym katalogu roboczym znajdują się dwa kluczowe pliki źródłowe:

1. `tinystories_model.pt` – plik zawierający stan i wyuczone wagi sieci neuronowej.
2. `tinystories_bpe.json` – konfiguracja oraz słownik wytrenowanego tokenizatora BPE.

#### Dostępne opcje i parametry wiersza poleceń (CLI)

Skrypt wykorzystuje moduł `argparse` do elastycznego sterowania procesem generowania opowiadań za pomocą następujących flag parametrów:

* `--prompt` (typ: `str`, domyślnie: `""`): Tekst początkowy lub fraza startowa, od której model rozpocznie układanie historii. W przypadku podania pustego ciągu znaków, model samodzielnie wylosuje pierwszy token.
* `--tokens` (typ: `int`, domyślnie: `200`): Maksymalna liczba nowych tokenów, które model wygeneruje autoregresyjnie (proces może zakończyć się wcześniej, jeśli wygenerowany zostanie token końca sekwencji `<eos>`).
* `--temp` (typ: `float`, domyślnie: `0.8`): Temperatura próbkowania (sampling temperature) skalująca wartości logits. Wyższa wartość zwiększa losowość i kreatywność tekstu, natomiast niższa wartość (bliższa 0) czyni go bardziej sztywnym i przewidywalnym.
* `--top_k` (typ: `int`, domyślnie: `10`): Ograniczenie próbkowania do $K$ najbardziej prawdopodobnych tokenów. Odcina ono tokeny o niskim prawdopodobieństwie przed zastosowaniem funkcji Softmax, zapobiegając powstawaniu rażących błędów gramatycznych.

#### Przykłady użycia w konsoli

```bash
python inference.py
python inference.py --prompt "One day"
python inference.py --prompt "One day, a big green dragon" --tokens 180 --temp 0.8 --top_k 10
```

Model był w stanie generować krótkie historie w stylu zbliżonym do zbioru TinyStories. Wygenerowane teksty zwykle zawierają prostego bohatera, miejsce akcji, wydarzenie oraz podstawowe zakończenie.

Przykład generacji bez promptu:

```text
Prompt: ""
```

Wygenerowany tekst:

> Once upon a time, there was a little girl named Lily. She loved to play with her dolls. One day, she asked her mom if she could play with her dolls, but her mom said no. Lily was sad. She went to her room and started to cry. Her mom noticed and asked, "Why can't you talk to her?"Lily replied, "I don't know, I can't find you. We can't find me."Her mom hugged her and said, "Don't worry, we'll find the lost puppy."They searched for the puppy in the garden, but they couldn't find him. Finally, they found the puppy hiding under a bush. Lily was so happy and said, "Thank you, Max. You're my best friend."From that day on, Lily and her mom were always friends and played together every day. And they always made sure to be kind to her and to always be kind to them

Ten wynik pokazuje, że model potrafi samodzielnie rozpocząć historię i utrzymać styl przypominający prostą bajkę. Pojawia się bohaterka, problem oraz próba jego rozwiązania. Jednocześnie można zauważyć pewne niespójności logiczne, na przykład przejście od zabawy lalkami do poszukiwania zgubionego szczeniaka.

Przykład generacji z krótkiego promptu:

```text
Prompt: "One day"
```

Wygenerowany tekst:

> One day, a little girl named Lily went to the park. She saw a big tree with lots of trees and flowers. She wanted to climb it. But it was very high for her to reach. Lily asked her mommy what it was, and her mommy said it was too high for her to reach. Lily tried to climb the tree, but it was too high for her to reach. She felt sad and didn't know what to do. Then, a kind man came to the park. He saw Lily and asked her what was wrong. Lily told him about her climbing the tree. The man said he was climbing trees, but Lily didn't want to climb the tree. She asked her daddy for help, and he gave her a ladder. He climbed up the ladder and climbed it. The man was very happy and thanked Lily for help. He said thank you to Lily and they both went home. Lily learned that being kind and compassionate is important, and that was a good way to

Ten przykład pokazuje, że model dobrze radzi sobie z promptami przypominającymi naturalny początek bajki. Kontynuacja zachowuje temat parku, drzewa i pomocy dziecku, chociaż w dalszej części pojawiają się powtórzenia, np. wielokrotne stwierdzenie, że drzewo było zbyt wysokie.

### 3.2. Eksperyment z temperaturą generacji

W pierwszym eksperymencie sprawdzono wpływ temperatury na generowany tekst. Użyto tego samego promptu:

```text
One day, a big green dragon
```

oraz ustawiono `top_k = 10` i `tokens = 180`. Testowano trzy wartości temperatury: `0.3`, `0.8` oraz `1.2`.

#### Temperatura 0.3

Wygenerowany tekst:

> One day, a big green dragon was walking in the forest. He was very brave and he wanted to explore the world. He asked his friends, "Can I go and explore the world?"His friends said, "Yes, you can go, but be careful. Don't be too close to the forest."The dragon was very brave and he was very excited. He ran around the forest and explored the forest. He saw many animals, but he was also very curious. He wanted to explore the world, so he decided to explore the forest. He found a big tree and started to climb up the tree. He was so excited he almost forgot the way he found. He was so excited he decided to explore the forest. He saw many trees and trees and trees. He was so happy he had found a new way to explore the world. He was so excited he couldn't wait to explore the forest. He

Dla temperatury `0.3` tekst był przewidywalny i stylistycznie stabilny, ale szybko pojawiły się powtórzenia. Model wielokrotnie wracał do motywów takich jak “forest”, “explore” oraz “excited”. Niska temperatura ogranicza losowość wyboru tokenów, przez co model częściej wybiera najbardziej prawdopodobne kontynuacje, ale może łatwiej wpadać w powtarzalne schematy.

#### Temperatura 0.8

Wygenerowany tekst:

> One day, a big green dragon was sitting on a tree. The dragon was very tall and tall. The dragon felt like a little bird. He was looking for the bird. The bird was hungry. He wanted to eat a snack. So, he went to the tree and sat down on a branch. The bird was happy too. He ate some food and ate the food. After eating, he became hungry. He ate the food and fell asleep. The end.

Dla temperatury `0.8` tekst był krótszy i bardziej zwarty. Nadal pojawiały się błędy oraz powtórzenia, np. “very tall and tall” oraz powtarzanie motywu jedzenia. Ta wartość temperatury dawała jednak rozsądny kompromis między spójnością a różnorodnością.

#### Temperatura 1.2

Wygenerowany tekst:

> One day, a big green dragon flew down and started moving his wings. He was flying very fast and the dragon was flying. The dragon flew over the sky and landed on the ground. He felt like he was flying, so he decided to make a sound on his wings. As he continued flying, he saw a big hill. The dragon flew down and landed near it. The giant had a beautiful view that he had never felt before. The dragon saw the dragon again. The dragon flew to take a closer look. It s wings started to free. After the dragon flew back, the star was safe. The dragon thanked the dragon for showing him the way to the dragon. The dragon flew back to him and flew back home with the big red bird in his beak. From that day on, the dragon was always there to help people and they were happy and safe.

Dla temperatury `1.2` historia była bardziej dynamiczna i różnorodna, ale pojawiło się więcej problemów logicznych. Model często powtarzał słowo “dragon” i tworzył mniej naturalne przejścia między wydarzeniami. Wyższa temperatura zwiększa kreatywność generacji, ale jednocześnie pogarsza kontrolę nad spójnością tekstu.

### 3.3. Eksperyment z parametrem top-k

W kolejnym eksperymencie sprawdzono wpływ parametru `top_k`, pozostawiając stałą temperaturę `0.8`. Użyto tego samego promptu:

```text
One day, a big green dragon
```

#### top_k = 1

Wygenerowany tekst:

> One day, a big green dragon was walking in the forest. He was very scared and didn't know what to do. Suddenly, he heard a voice. It was a little bird. The bird was scared and flew away. The dragon was very scared. He tried to fly away, but the bird was too fast. The dragon was very brave and flew away. The dragon was sad and scared. He wished he could fly away. The dragon flew away and never came back.

Dla `top_k = 1` model wybierał zawsze najbardziej prawdopodobny token. Tekst był prosty i dość spójny, ale mało zróżnicowany. Historia zakończyła się szybko i miała przewidywalny przebieg.

#### top_k = 5

Wygenerowany tekst:

> One day, a big green dragon was walking in the forest. The dragon saw many things and wanted to see who could help. He asked his friends to help him get the things he needed. They all worked together and soon the dragon was ready to help. The dragon was very happy and said, "Thank you for helping me."The dragon was very happy and thanked the bird for helping him. They became great friends and played together every day. From that day on, the dragon always remembered to be careful when he needed help when he needed.

Dla `top_k = 5` tekst zachował temat smoka i lasu, a jednocześnie był nieco bardziej zróżnicowany niż dla `top_k = 1`. Nadal pojawiały się jednak powtórzenia, szczególnie motyw pomocy oraz powtarzające się zdania o zadowoleniu smoka.

#### top_k = 50

Wygenerowany tekst:

> One day, a big green dragon was walking out the door. He was feeling curious and wanted to find out what was inside. He ca ut ious ly stepped inside the door, looking over at the sky and saw an adorable dragon. He quickly climbed up and carefully opened it. He opened the door and inside was a beautiful little bunny. The bunny was so excited and he quickly ran inside to tell his mum about the dragon. His mum smiled and said, "Let's go look for you, but we have to hurry now."She grabbed a stick and the dragon was very excited. She opened the door and the dragon was happy. She couldn't wait to explore it.

Dla `top_k = 50` model miał większą swobodę wyboru tokenów. Wynik był bardziej różnorodny, ale pojawiły się wyraźne artefakty tokenizacji, np. “ca ut ious ly”. Tekst stał się mniej przewidywalny i mniej stabilny. Pokazuje to, że zbyt duże `top_k` może zwiększać kreatywność, ale jednocześnie podnosi ryzyko wyboru mniej trafnych tokenów.

### 3.4. Eksperyment z różnymi typami promptów

Następnie sprawdzono, jak model reaguje na różne rodzaje promptów. Najlepsze wyniki uzyskano dla promptów przypominających naturalny początek historii.

#### Naturalny prompt bajkowy

Prompt:

```text
Once upon a time, there was a little girl named Lily
```

Wygenerowany tekst:

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and her friends would always play together. One day, Lily's mom asked her to clean her room. Lily didn't want to clean her room, but she was having so much fun. Lily's mom explained that it's important to clean her room, but it's important to clean up the mess. Lily didn't want to clean her room, but she remembered her mom's words. She decided to clean up the messy room and make sure she was clean. Lily put on her favorite pa jam as and made a mess. She felt sad and started to clean her room. From that day on, Lily knew that she could help her room clean and tidy up the mess.

W tym przypadku model dobrze kontynuował podany początek. Historia zachowała bohaterkę Lily oraz prosty dziecięcy kontekst. Pojawił się typowy dla TinyStories motyw nauki dobrego zachowania, czyli sprzątania pokoju.

#### Prompt przypominający instrukcję

Prompt:

```text
Write a story about flowers
```

Wygenerowany tekst:

> W r ite a story about flowers and butterflies. She wanted to be a flower. She asked her mom, "Mom, can we help her?"Her mom said, "Yes, we can be a queen. But it is too far away."W r ite a s igh and said, "Don't worry, I'll help you get a flower. I will help you see it and see what happ ens."So she carefully lifted the flower and gave it to her mom. Her mom smiled and said, "That's a good idea, W r ince, I can help you."

Model nie potraktował tego promptu jako polecenia, tylko jako zwykły początek tekstu do kontynuacji. Dodatkowo słowo “Write” zostało rozbite jako “W r ite”, co wskazuje na problem tokenizacji. Wynik pokazuje, że model nie jest modelem instrukcyjnym ani chatbotem. Nie został wytrenowany do wykonywania poleceń, tylko do kontynuowania tekstu.

#### Prompt spoza domeny danych treningowych

Prompt:

```text
The quantum computer calculated
```

Wygenerowany tekst:

> The qu ant um computer cal c ul ated in the living room. One day, Lily's little brother came over and saw that he was feeling sleepy. She asked her what was wrong, and her brother said it was okay. Lily told him that the tele vis ion was just for her. She said it was okay and they went to the c ul ture. They sat by the c ul ture, and the tele vis ion was still comfortable. Lily felt better and went to bed. She felt comfortable and ready to sleep. The next day, Lily woke up feeling much better. She was happy that she could help her brother feel better, and she felt better. She went to bed feeling much better. The end.

W tym przykładzie słowa związane z technologią, takie jak “quantum”, “computer” i “calculated”, zostały rozbite na mniejsze fragmenty. Model szybko przeszedł z tematu komputera kwantowego do typowej dziecięcej historii o Lily i rodzinie. Oznacza to, że model najlepiej działa dla promptów podobnych do danych treningowych, a gorzej dla tematów technicznych lub rzadko występujących w zbiorze TinyStories.

### 3.5. Obserwacje

Przeprowadzone eksperymenty pokazują, że model nauczył się ogólnego stylu prostych historii dziecięcych. Potrafi generować krótkie zdania, używać prostego słownictwa, wprowadzać bohaterów oraz tworzyć podstawową strukturę narracyjną. Najlepsze wyniki uzyskiwano dla promptów, które brzmiały jak naturalny początek bajki, np. “One day” albo “Once upon a time”.

Jednocześnie model ma kilka widocznych ograniczeń. Najczęściej pojawiają się powtórzenia tych samych słów, motywów i konstrukcji zdaniowych. Widać to szczególnie przy niskiej temperaturze oraz przy dłuższej generacji. Model potrafi utrzymywać lokalną spójność tekstu, ale nie zawsze zachowuje globalny plan historii.

Drugim istotnym ograniczeniem są artefakty tokenizacji. W przypadku mniej typowych słów lub promptów spoza domeny treningowej pojawiały się fragmenty takie jak “W r ite”, “qu ant um”, “cal c ul ated”, “tele vis ion” oraz “ca ut ious ly”. Oznacza to, że jakość działania modelu zależy nie tylko od architektury, ale również od jakości tokenizatora i podobieństwa promptu do danych treningowych.

Eksperymenty z parametrami generacji pokazały, że temperatura i `top_k` mają duży wpływ na wynik. Niska temperatura daje bardziej przewidywalne, ale często bardziej powtarzalne teksty. Wyższa temperatura zwiększa różnorodność, ale pogarsza spójność. Małe `top_k` ogranicza kreatywność, natomiast duże `top_k` może prowadzić do bardziej zaskakujących, ale mniej stabilnych wyników.

Ogólnie wyniki można uznać za satysfakcjonujące jak na mały model językowy. System nie osiąga jakości dużych modeli konwersacyjnych, ale dobrze pokazuje podstawowe mechanizmy generowania tekstu: kontynuowanie promptu, wpływ parametrów próbkowania oraz ograniczenia wynikające z rozmiaru modelu i danych treningowych.


## 4. Wnioski

Jednym z najważniejszych wniosków z projektu jest to, że do generowania prostych historii nie jest konieczny bardzo duży model językowy. Udało się pokazać, że model mający około 4 miliony parametrów jest w stanie nauczyć się stylu zbioru TinyStories i generować krótkie, w dużej mierze zrozumiałe historie. Oczywiście taki model ma wyraźne ograniczenia: częściej powtarza słowa, traci spójność przy dłuższej generacji i gorzej radzi sobie z promptami spoza domeny danych treningowych. Mimo to wyniki pokazują, że nawet niewielki Transformer może skutecznie uchwycić podstawowe schematy narracyjne, takie jak bohater, proste wydarzenie, problem i zakończenie historii.


Najważniejszym wnioskiem jest to, że model działa najlepiej wtedy, gdy prompt jest podobny do danych treningowych. Prompty takie jak “One day” albo “Once upon a time” prowadziły do bardziej naturalnych i spójnych historii. Z kolei prompty techniczne, nietypowe lub przypominające instrukcje, np. “Write a story about flowers” albo “The quantum computer calculated”, powodowały spadek jakości generacji. Model nie został wytrenowany jako chatbot ani model instrukcyjny, dlatego nie interpretuje promptu jako polecenia, tylko jako początek tekstu do kontynuacji.

Eksperymenty pokazały również istotny wpływ parametrów generacji. Niska temperatura dawała bardziej przewidywalne teksty, ale zwiększała ryzyko powtórzeń. Wyższa temperatura pozwalała uzyskać bardziej różnorodne historie, jednak częściej prowadziła do błędów logicznych i mniej stabilnej narracji. Podobnie parametr `top_k` wpływał na balans między przewidywalnością a kreatywnością: małe wartości ograniczały różnorodność, natomiast duże wartości zwiększały ryzyko mniej trafnych kontynuacji.

Ważnym ograniczeniem okazała się tokenizacja. Słowa rzadkie lub spoza domeny danych treningowych były często rozbijane na mniejsze fragmenty, np. “qu ant um”, “cal c ul ated”, “W r ite” czy “g mail”. Model próbował następnie dopasować takie fragmenty do znanych schematów z danych treningowych, czasami traktując je jak nazwy własne, imiona, obiekty lub elementy świata przedstawionego. Pokazuje to, że jakość tokenizatora oraz podobieństwo promptu do danych treningowych mają duży wpływ na końcowy wynik.

Mimo widocznych ograniczeń projekt osiągnął swój główny cel. Udało się stworzyć kompletny pipeline generowania tekstu: od tokenizacji, przez wczytanie modelu, aż po autoregresyjną generację i dekodowanie wyniku. Model nie osiąga jakości dużych modeli konwersacyjnych, ale dobrze demonstruje podstawowe mechanizmy działania małych modeli językowych oraz pokazuje, jak parametry próbkowania, tokenizacja i dane treningowe wpływają na generowany tekst.

## 5. Bibliografia
### Artykuły naukowe
* Eldan, R., & Li, Y. (2023). TinyStories: How Small Can Language Models Be and Still Speak Coherent English? arXiv preprint arXiv:2305.07759.
### Zbiory danych
* Zbiór danych TinyStories (Hugging Face): roneneldan/TinyStories
### Biblioteki
* PyTorch
* Hugging Face Tokenizers
* Hugging Face Datasets


