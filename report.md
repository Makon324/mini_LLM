## 3. Wyniki

### 3.1. Prezentacja działającego systemu

W wyniku realizacji projektu uzyskaliśmy działający mały model językowy zdolny do generowania krótkich historii w języku angielskim. Model działa autoregresyjnie: otrzymuje początkowy prompt, przewiduje następny token, dołącza go do aktualnej sekwencji i powtarza ten proces aż do wygenerowania zadanej liczby tokenów lub zakończenia sekwencji.

System można uruchomić z poziomu wiersza poleceń za pomocą skryptu inferencyjnego. Skrypt wczytuje tokenizator oraz wytrenowane wagi modelu, koduje prompt, generuje nowe tokeny, dekoduje je z powrotem do tekstu, a następnie wykonuje podstawowe przetwarzanie końcowe, między innymi usuwanie zbędnych spacji przed znakami interpunkcyjnymi.

Przykładowe użycie:

```bash
python inference.py
python inference.py --prompt "One day"
python inference.py --prompt "One day, a big green dragon" --tokens 180 --temp 0.8 --top_k 10
```

Model był w stanie generować krótkie historie w stylu zbliżonym do zbioru TinyStories. Wygenerowane teksty zwykle zawierają prostego bohatera, miejsce akcji, wydarzenie oraz podstawowe zakończenie. Przykładowo, dla generacji bez promptu model utworzył następujący fragment:

> Once upon a time, there was a little girl named Lily. She loved to play with her dolls. One day, she asked her mom if she could play with her dolls, but her mom said no. Lily was sad. She went to her room and started to cry. (...)

Tekst zachowuje prosty, dziecięcy styl narracji. Pojawia się bohaterka, problem oraz próba jego rozwiązania. Jednocześnie można zauważyć pewne niespójności logiczne, na przykład przejście od zabawy lalkami do poszukiwania zgubionego szczeniaka.

Dla promptu:

```text
One day
```

model wygenerował historię rozpoczynającą się następująco:

> One day, a little girl named Lily went to the park. She saw a big tree with lots of trees and flowers. She wanted to climb it. But it was very high for her to reach. (...)

Ten przykład pokazuje, że model dobrze radzi sobie z promptami przypominającymi naturalny początek bajki. Kontynuacja zachowuje temat parku, drzewa i pomocy dziecku, chociaż w dalszej części pojawiają się powtórzenia, np. wielokrotne stwierdzenie, że drzewo było zbyt wysokie.

### 3.2. Eksperyment z temperaturą generacji

W pierwszym eksperymencie sprawdzono wpływ temperatury na generowany tekst. Użyto tego samego promptu:

```text
One day, a big green dragon
```

oraz ustawiono `top_k = 10` i `tokens = 180`. Testowano trzy wartości temperatury: `0.3`, `0.8` oraz `1.2`.

Dla temperatury `0.3` model wygenerował tekst bardziej przewidywalny i stabilny:

> One day, a big green dragon was walking in the forest. He was very brave and he wanted to explore the world. (...)

Wynik był dość spójny stylistycznie, ale szybko pojawiły się powtórzenia. Model wielokrotnie wracał do słów i motywów takich jak “forest”, “explore” oraz “excited”. Niska temperatura ogranicza losowość wyboru tokenów, przez co model częściej wybiera najbardziej prawdopodobne kontynuacje, ale może łatwiej wpadać w powtarzalne schematy.

Dla temperatury `0.8` model wygenerował krótszą, bardziej zwartą historię:

> One day, a big green dragon was sitting on a tree. The dragon was very tall and tall. The dragon felt like a little bird. (...)

Tekst był mniej powtarzalny niż przy temperaturze `0.3`, ale nadal zawierał pewne problemy, np. powtórzenie “very tall and tall” oraz uproszczoną logikę historii. Ta wartość temperatury dawała rozsądny kompromis między spójnością a różnorodnością.

Dla temperatury `1.2` model wygenerował bardziej dynamiczny i kreatywny tekst:

> One day, a big green dragon flew down and started moving his wings. He was flying very fast and the dragon was flying. (...)

W tym przypadku historia była bardziej zróżnicowana, ale pojawiło się więcej problemów logicznych. Model częściej powtarzał słowo “dragon” i tworzył mniej naturalne przejścia między zdarzeniami. Wyższa temperatura zwiększa różnorodność generacji, ale jednocześnie pogarsza kontrolę nad spójnością tekstu.

### 3.3. Eksperyment z parametrem top-k

W kolejnym eksperymencie sprawdzono wpływ parametru `top_k`, pozostawiając stałą temperaturę `0.8`. Użyto tego samego promptu:

```text
One day, a big green dragon
```

Dla `top_k = 1` model wybierał zawsze najbardziej prawdopodobny token. Wygenerowany tekst był prosty i dość spójny:

> One day, a big green dragon was walking in the forest. He was very scared and didn't know what to do. (...)

Taki sposób generacji daje stabilny wynik, ale ogranicza kreatywność. Historia była krótka, przewidywalna i zakończyła się dość szybko.

Dla `top_k = 5` model uzyskał bardziej naturalny kompromis:

> One day, a big green dragon was walking in the forest. The dragon saw many things and wanted to see who could help. (...)

Tekst zachował temat smoka i lasu, a jednocześnie był trochę bardziej zróżnicowany niż dla `top_k = 1`. Nadal pojawiały się jednak powtórzenia, na przykład wielokrotne użycie motywu pomocy.

Dla `top_k = 50` model miał większą swobodę wyboru tokenów:

> One day, a big green dragon was walking out the door. He was feeling curious and wanted to find out what was inside. He ca ut ious ly stepped inside the door (...)

Wynik był bardziej różnorodny, ale pojawiły się wyraźne artefakty tokenizacji, np. “ca ut ious ly”. Tekst stał się mniej przewidywalny i mniej stabilny. Pokazuje to, że zbyt duże `top_k` może zwiększać kreatywność, ale jednocześnie podnosi ryzyko wyboru mniej trafnych tokenów.

### 3.4. Eksperyment z różnymi typami promptów

Następnie sprawdzono, jak model reaguje na różne rodzaje promptów. Najlepsze wyniki uzyskano dla promptów przypominających naturalny początek historii, np.:

```text
Once upon a time, there was a little girl named Lily
```

Model wygenerował kontynuację:

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and her friends would always play together. One day, Lily's mom asked her to clean her room. (...)

W tym przypadku model dobrze kontynuował podany początek. Historia zachowała bohaterkę Lily oraz prosty dziecięcy kontekst. Pojawił się typowy dla TinyStories motyw nauki dobrego zachowania, czyli sprzątania pokoju.

Gorsze wyniki pojawiły się dla promptów przypominających instrukcję, np.:

```text
Write a story about flowers
```

Wygenerowany tekst rozpoczął się od:

> W r ite a story about flowers and butterflies. She wanted to be a flower. (...)

Model nie potraktował tego promptu jako polecenia, tylko jako zwykły początek tekstu do kontynuacji. Dodatkowo słowo “Write” zostało rozbite jako “W r ite”, co wskazuje na problem tokenizacji. Wynik pokazuje, że model nie jest modelem instrukcyjnym ani chatbotem. Nie został wytrenowany do wykonywania poleceń, tylko do kontynuowania tekstu.

Podobny problem wystąpił dla promptu spoza domeny danych treningowych:

```text
The quantum computer calculated
```

Model wygenerował:

> The qu ant um computer cal c ul ated in the living room. One day, Lily's little brother came over (...)

W tym przykładzie słowa związane z technologią, takie jak “quantum”, “computer” i “calculated”, zostały rozbite na mniejsze fragmenty. Model szybko przeszedł z tematu komputera kwantowego do typowej dziecięcej historii o Lily i rodzinie. Oznacza to, że model najlepiej działa dla promptów podobnych do danych treningowych, a gorzej dla tematów technicznych lub rzadko występujących w zbiorze TinyStories.

### 3.5. Obserwacje

Przeprowadzone eksperymenty pokazują, że model nauczył się ogólnego stylu prostych historii dziecięcych. Potrafi generować krótkie zdania, używać prostego słownictwa, wprowadzać bohaterów oraz tworzyć podstawową strukturę narracyjną. Najlepsze wyniki uzyskiwano dla promptów, które brzmiały jak naturalny początek bajki, np. “One day” albo “Once upon a time”.

Jednocześnie model ma kilka widocznych ograniczeń. Najczęściej pojawiają się powtórzenia tych samych słów, motywów i konstrukcji zdaniowych. Widać to szczególnie przy niskiej temperaturze oraz przy dłuższej generacji. Model potrafi utrzymywać lokalną spójność tekstu, ale nie zawsze zachowuje globalny plan historii.

Eksperymenty z parametrami generacji pokazały, że temperatura i `top_k` mają duży wpływ na wynik. Niska temperatura daje bardziej przewidywalne, ale często bardziej powtarzalne teksty. Wyższa temperatura zwiększa różnorodność, ale pogarsza spójność. Małe `top_k` ogranicza kreatywność, natomiast duże `top_k` może prowadzić do bardziej zaskakujących, ale mniej stabilnych wyników.

Istotnym ograniczeniem okazały się również artefakty tokenizacji. W przypadku mniej typowych słów lub promptów spoza domeny treningowej pojawiały się fragmenty takie jak “W r ite”, “qu ant um”, “cal c ul ated”, “tele vis ion”, “g mail” oraz “ca ut ious ly”. Oznacza to, że jakość działania modelu zależy nie tylko od architektury, ale również od jakości tokenizatora i podobieństwa promptu do danych treningowych.

Dodatkowo zauważono, że gdy w promptach pojawiają się słowa rzadkie lub słabo reprezentowane w danych treningowych, model po rozbiciu ich na mniejsze fragmenty często próbuje dopasować te fragmenty do znanych schematów narracyjnych. Przykładowo prompt “gmail” został rozbity na “g mail”, po czym model zaczął generować historię związaną z listem, pocztą i słowem “mailer”. Podobnie słowa techniczne, takie jak “quantum” czy “calculated”, nie były traktowane jako pojęcia techniczne, tylko jako fragmenty tekstu, które model próbował wpasować w prostą historię dziecięcą.

Można również zauważyć, że nietypowe lub rozbite słowa bywają przez model traktowane jak nazwy własne, imiona, obiekty albo elementy świata przedstawionego. Wynika to prawdopodobnie z faktu, że w danych treningowych często pojawiają się proste historie zaczynające się od bohatera, przedmiotu lub miejsca. Model nie rozumie więc znaczenia rzadkiego słowa w taki sposób jak człowiek, tylko dopasowuje jego fragmenty do wzorców językowych poznanych podczas treningu.

Ta obserwacja pokazuje, że model najlepiej działa dla promptów podobnych do danych treningowych. Jeśli prompt zawiera słownictwo spoza tej domeny, szczególnie techniczne lub nietypowe, jakość generacji spada. Model może wtedy nie tylko rozbić słowa na fragmenty tokenów, ale też błędnie nadać im rolę w historii, np. potraktować je jak bohaterów, imiona lub obiekty występujące w bajce.

Ogólnie wyniki można uznać za satysfakcjonujące jak na mały model językowy. System nie osiąga jakości dużych modeli konwersacyjnych, ale dobrze pokazuje podstawowe mechanizmy generowania tekstu: kontynuowanie promptu, wpływ parametrów próbkowania oraz ograniczenia wynikające z rozmiaru modelu, tokenizacji i danych treningowych.


