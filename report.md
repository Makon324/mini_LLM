Jasne, oto profesjonalny raport w formacie Markdown, przygotowany na podstawie dostarczonych plików z kodem oraz dyskusji z Discorda. Raport łączy formalną strukturę z wnioskami, które wyciągnęliście podczas testowania modelu.

---

## Raport z Projektu: Model Językowy TinyStories

### Introduction

**Problem description**
Współczesne modele językowe (LLM) osiągają imponujące wyniki w generowaniu tekstu, jednak ich trenowanie wymaga ogromnych zasobów obliczeniowych oraz potężnych zbiorów danych. Głównym problemem jest zbadanie, czy możliwe jest stworzenie spójnie działającego, miniaturowego modelu językowego na ograniczonych zasobach, który potrafiłby generować logiczne, choć uproszczone historie w języku angielskim.

**Project goal**
Celem projektu było zaimplementowanie, wytrenowanie i przetestowanie małego modelu autoregresyjnego opartego na architekturze Transformer. Model o wielkości około 4 milionów parametrów miał zostać nauczony na specyficznym, uproszczonym zbiorze danych, aby z powodzeniem generować krótkie opowiadania i reagować na podane przez użytkownika prompty.

---

### Methods and Data

**Method description**
Zastosowano architekturę Transformer z mechanizmem przyczynowej uwagi (Causal Self-Attention), która jest standardem dla modeli generujących tekst (np. seria GPT). Sieć działa w sposób autoregresyjny, co oznacza, że na podstawie podanego kontekstu przewiduje najbardziej prawdopodobny następny token (słowo lub jego fragment). Aby sieć mogła analizować pozycję słów, użyto zarówno zanurzeń tokenów (Token Embeddings), jak i zanurzeń pozycyjnych (Position Embeddings).

Zastosowany mechanizm atencji można opisać formalnym wzorem matematycznym, w którym zapytania ($Q$), klucze ($K$) oraz wartości ($V$) są wykorzystywane do obliczenia wag uwagi dla każdego tokenu:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**State-of-the-art summary**
Współczesne modele takie jak GPT-4 czy LLaMA polegają na miliardach parametrów. Jednak badania nad modelami typu "TinyStories" udowadniają, że zastosowanie odpowiednio przefiltrowanego, syntetycznego zbioru danych (złożonego z prostych słów dostosowanych do poziomu kilkulatków) pozwala drastycznie zmniejszyć architekturę sieci. Dzięki temu, nawet sieci o ułamku wielkości tradycyjnych LLM-ów potrafią przyswoić poprawną gramatykę i podstawy rozumowania przyczynowo-skutkowego.

**Technology description**
Całość rozwiązania została zaimplementowana w języku Python przy użyciu biblioteki PyTorch. Skrypt odpowiedzialny za wnioskowanie pozwala na dynamiczne próbkowanie z wykorzystaniem parametrów takich jak `temperature` oraz `top_k`.

Poniższa tabela przedstawia kluczowe hiperparametry wytrenowanego modelu:

| Hiperparametr | Wartość | Opis |
| --- | --- | --- |
| **Rozmiar słownika (VOCAB_SIZE)** | 4096 | Liczba unikalnych tokenów |
| **Długość kontekstu (CONTEXT_LENGTH)** | 256 | Maksymalna liczba tokenów przetwarzanych naraz |
| **Wymiar embeddingu (N_EMBD)** | 256 | Rozmiar wektora reprezentującego token |
| **Liczba warstw (N_LAYER)** | 4 | Liczba bloków Transformera |
| **Głowy atencji (N_HEAD)** | 8 | Liczba niezależnych mechanizmów atencji (32 wymiary na głowę) |
| **Parametry trenowania** | 3 Epoki | Trenowane przez 2 epoki z dodatkową 1 epoką (Learning Rate: 5e-4) |

**Data description**
Wykorzystano zbiór danych `roneneldan/TinyStories` pobrany z platformy Hugging Face. Zbiór ten zawiera wygenerowane opowiadania dla małych dzieci. Przetwarzanie danych obejmowało stworzenie dedykowanego tokenizatora BPE (Byte-Pair Encoding), który analizował zbiór partiami po 10 000 historii. Tokenizator wzbogacono o specjalne tokeny `<bos>` (początek zdania) oraz `<eos>` (koniec zdania).

---

### Results

**Experimental results or working system presentation**
Model po wytrenowaniu został pomyślnie uruchomiony w trybie wnioskowania (na CPU). Okazał się zdolny do samodzielnego generowania wielozdaniowych historii na podstawie krótkich promptów. Przykład wygenerowanej historii dla promptu wprowadzającego:

> **Prompt:** "One day, a big green dragon"
> **Wygenerowany tekst:** "One day, a big green dragon named Tom went for a walk. Tom liked to see the big green mountain in the distance. He saw many animals like trees, birds, and birds. He felt happy and free. One day, Tom went to the forest and said, "I want to see the animals. I want to see them up close!" The bird said, "I want to go see them. They are too big and too heavy." Tom and the bird became friends and played together every day."

**Observations**
Na podstawie analizy działania modelu wyciągnięto następujące wnioski:

* **Spójność postaci i gramatyka:** Model (posiadający ok. 4 mln parametrów) radzi sobie zadziwiająco dobrze z zachowaniem ciągłości postaci (np. smok Tom, kot, dziewczynka Lily). Generuje poprawne gramatycznie, choć bardzo proste, zdania w języku angielskim.
* **Repetytywność (Halucynacje):** Zauważono skłonność do zapętlania się i powtarzania fraz w przypadku niektórych promptów (np. trzykrotne powtórzenie *"The dog is gone."* lub *"They have no more flowers."*).
* **Problemy z Out-of-Distribution (OOD):** Nietypowe wielkie litery stanowią wyzwanie. Użytkownik wprowadzający prompt *"Story about flowers"* sprawił, że tokenizator podzielił słowo na dwie części *"S tory"*, ponieważ wyraz "Story" pisany wielką literą pojawiał się zbyt rzadko w korpusie treningowym, aby stanowić niezależny token. Model nie potrafił poprawnie uwarunkować na tym swojej uwagi.
* **Post-processing interpunkcji:** Aby poprawić jakość i czytelność tekstu wychodzącego, zastosowano "twarde" reguły w skrypcie inferencyjnym, programowo usuwając spacje przed znakami interpunkcyjnymi (np. przecinkami i kropkami), które wynikały ze specyfiki działania zastosowanego tokenizatora.

---

### Conclusion

**Key takeaways and insights**
Eksperyment zakończył się sukcesem i dowodzi, że budowa własnego modelu LLM od zera jest jak najbardziej wykonalna na domowym sprzęcie studenckim. Głównym wnioskiem płynącym z projektu jest fakt, że rozmiar modelu (liczba parametrów) ma mniejsze znaczenie niż jakość i struktura zbioru uczącego w kontekście generowania podstawowych, gramatycznie poprawnych zdań. Aby uzyskać najlepsze rezultaty, należy stosować odpowiedni *prompt engineering* (np. wprowadzenie typu *"Once upon a time"* lub *"One day"*), co wprowadza model w odpowiednią przestrzeń ukrytą (latent space), zgodną ze zbiorem TinyStories. Ewentualne ulepszenia na przyszłość powinny obejmować poprawienie procesów tokenizacji oraz powiększenie rozmiaru słownika (VOCAB_SIZE) w celu lepszej obsługi kapitalizacji.

**References**

* Zbiór danych: HuggingFace Datasets (`roneneldan/TinyStories`)
* Implementacja modelu: Biblioteka `torch` (PyTorch) oraz `torch.nn`
* Narzędzia tekstowe: `tokenizers` (Hugging Face)
* Pliki projektowe:
* Plik treningowy środowiska Jupyter: `proj3.ipynb`
* Skrypt do wnioskowania i architektura modelu: `inference.py`
