# Knowledge Base (example)

## Example marble head (Παράδειγμα κεφαλής)
- URI: https://example.org/collection/item/0001

**Type:** Sculpture (Γλυπτό) / Archaeological object (Αρχαιολογικό αντικείμενο)
**Material:** Marble (Μάρμαρο)
**Time Period:** 101 AD - 200 AD [Roman Period (Ρωμαϊκή περίοδος)]
**Location Found:** Example City (Παράδειγμα Πόλη)
**Dimensions:** Height: 25.0 cm

**Description:**
A short example description of the artifact. The RAG path splits this Markdown on
`##` headers into chunks, then indexes each chunk with FAISS + BM25.

**Subjects & Themes:**
Antiquity (Αρχαιότητα), Sculpture (Γλυπτική)

---

## Example portable icon (Παράδειγμα εικόνας)
- URI: https://example.org/collection/item/0002

**Type:** Portable icon (Φορητή Εικόνα)
**Material:** Wood (Ξύλο)
**Time Period:** 1826 AD - 1849 AD
**Location Found:** Example Region

**Description:**
Each artifact is one `##` section. Keep the same URI/id you use in the JSONL file
and in the Neo4j graph so the KG and RAG paths line up.
