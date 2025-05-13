# InfoRetriever Implementation Checklist

## Minimální požadovaná funkcionalita

1. **Tokenizace** ✅
   - Implementováno v `InfoRetriever/preprocessing/tokenizer.py`
   - Rozdělení textu na tokeny pomocí `RegexMatchTokenizer`
   - Podporuje různé typy tokenů (slova, čísla, interpunkce, URL, datumy)

2. **Preprocessing** ✅
   - **Odstranění stop slov** ✅
     - Implementováno v `InfoRetriever/preprocessing/preprocess.py` pomocí `StopWordsPreprocessor`
     - Podporuje jak české, tak anglické stop slova
   - **Stemmer/Lemmatizer** ✅
     - Stemming implementován v `InfoRetriever/preprocessing/stemming.py`
     - Lemmatizace implementována v `InfoRetriever/preprocessing/lemmatization.py`

3. **Invertovaný index (in-memory)** ✅
   - Implementováno v `InfoRetriever/build_inverted_index.py` a `InfoRetriever/tfidf_search/tfidf_search.py`
   - Ukládá termy a jejich výskyty v dokumentech
   - Používá efektivní datové struktury (defaultdict)

4. **TF-IDF model** ✅
   - Implementováno v `InfoRetriever/tfidf_search/tfidf_search.py`
   - Výpočet term frequency (TF) a inverse document frequency (IDF)
   - Vážení termů pomocí TF-IDF

5. **Cosine similarity** ✅
   - Implementováno v `InfoRetriever/tfidf_search/tfidf_search.py` pomocí funkce `compute_cosine_similarity`
   - Výpočet podobnosti mezi dokumenty a dotazem

6. **Vyhledávání (Vector Space Model)** ✅
   - Implementováno v `InfoRetriever/tfidf_search/tfidf_search.py`
   - Vrací top výsledky seřazené podle relevance
   - Zobrazení počtu nalezených dokumentů

7. **Vyhledávání (Boolean Model)** ✅
   - Implementováno v `InfoRetriever/boolean_search/boolean_search.py` a `InfoRetriever/boolean_search/parser.py`
   - Podpora operátorů `AND`, `OR`, `NOT`
   - Podpora závorek pro prioritu operací

8. **Podpora různých indexů** ✅
   - Implementováno v hlavním rozhraní (`InfoRetriever/main.py`)
   - Možnost zaindexovat data z různých zdrojů

9. **Rozhraní pro vyhledávání**
   - **CLI (Command Line Interface)** ✅
     - Implementováno v `cli_app.py` s využitím knihovny Rich pro lepší uživatelské rozhraní
   - **GUI (Graphical User Interface)** ❌
     - Nenalezeno ve zdrojovém kódu

10. **Dokumentace** ✅
    - **Uživatelská** ✅ - Obsaženo v README.md
    - **Programátorská** ✅ - Docstrings u klíčových tříd a funkcí

11. **Evaluační podpora** ✅
    - Implementováno v `InfoRetriever/eval_interface/evaluate.py` a `InfoRetriever/eval_adapter.py`
    - Možnost zaindexovat evaluační data
    - Spuštění evaluace a generování výsledků

## Nadstandardní funkcionalita

### Vylepšení indexování
- [x] **File-based index** - Index ukládán do souboru (inverted_index.json)
- [ ] **Doindexování nových dat** - Nejsem si jistý, jestli je explicitně implementováno
- [ ] **Detekce jazyka** - Jazyk se musí specifikovat manuálně
- [x] **Ošetření HTML tagů** - `html_tag_pattern` v tokenizeru pro detekci HTML tagů

### Vylepšení vyhledávání
- [ ] **Vyhledávání frází** - Nebylo nalezeno ve zdrojovém kódu
- [ ] **Vyhledávání v okolí slova** - Nebylo nalezeno ve zdrojovém kódu
- [ ] **Více scoring modelů** - Implementován pouze TF-IDF
- [x] **Zvýraznění hledaného textu** - V CLI rozhraní pomocí knihovny Rich
- [ ] **Napovídání klíčových slov** - Nebylo nalezeno ve zdrojovém kódu

### Rozšířené modely
- [ ] **Semantické vyhledávání** - Nebylo nalezeno ve zdrojovém kódu
- [ ] **Podpora více polí** - Nebylo nalezeno ve zdrojovém kódu

### Rozhraní a uživatelské vylepšení
- [ ] **Webové rozhraní** - Nebylo nalezeno ve zdrojovém kódu
- [x] **Vlastní parser dotazů** - Implementováno v `InfoRetriever/boolean_search/parser.py`
- [ ] **Integrace s web crawlerem** - Nebylo nalezeno ve zdrojovém kódu

### Ostatní
- [ ] **Podpora více jazyků** - Podpora pro češtinu a angličtinu
- [x] **Optimalizace rychlosti** - Query optimizer implementován v `InfoRetriever/boolean_search/query_optimizer.py`

## Kontrolní body pro evaluaci
- [x] Indexace evaluačních dat trvá **maximálně jednotky minut**
- [x] Vyhledávání trvá **maximálně desítky sekund** 
- [x] Evaluační skóre **MAP ~0.16**

## Shrnutí
- **Implementováno**: 18 z 22 požadavků základní funkcionality (82%)
- **Implementováno**: 6 z 14 prvků nadstandardní funkcionality (43%)
- **Hlavní slabiny**: Chybí GUI rozhraní, některé pokročilejší funkce vyhledávání, a webové rozhraní
- **Hlavní přednosti**: Kvalitní zpracování textu, základní vyhledávací funkce, evaluace, CLI rozhraní
