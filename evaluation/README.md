# CultureBot expert evaluation

This page documents the expert evaluation reported in the CultureBot paper. It
is intended as a stable, citable description of the study design, with
particular emphasis on the personas used to represent realistic cultural
heritage information needs.

## Evaluation at a glance

Six cultural heritage specialists helped define the evaluation scenarios and
assessed the systems' answers. The study used:

- **12 expert-defined personas** spanning professional, educational, creative,
  and public-facing uses of cultural heritage collections;
- **55 expert-authored queries** covering factual, spatial, visual,
  comparative, thematic, and multi-step information needs;
- **four systems**: ChatGPT, Gemini, CultureBot without its knowledge graph,
  and the full graph-guided CultureBot
- **250 evaluation sessions**, with every answer assessed by at least two
  experts through a purpose-built annotation interface.

The personas are scenario-based user roles, not profiles of the participating
experts. They were used to make the query set reflect concrete research and
discovery tasks instead of generic question answering.

## Personas and use cases

| ID | Persona | Evaluation focus |
|---|---|---|
| P1 | Museum curator | Exhibition-oriented exploration of ancient Greek pottery for children, adults, and non-specialist audiences. |
| P2 | Archaeology student - material and context research | Burial contexts, material categories, and object inventories, with an emphasis on obsidian finds. |
| P3 | Archaeology student - visual research | Image-oriented search by artefact category and period across Greek cultural heritage collections. |
| P4 | Ethics and repatriation officer | Sensitive questions about human remains, storage, display, worship contexts, and ethical documentation. |
| P5 | Contemporary visual artist | Feminist and iconographic exploration of female sanctity, Byzantine imagery, and representations of women. |
| P6 | High-school student | School-oriented questions about historical artefacts, ancient imagery, weapons, mythology, and museum objects. |
| P7 | Craftsperson designing for museum retail | Search for historical forms, patterns, textures, and colour palettes that can inform contemporary objects. |
| P8 | Cultural journalist | Object-based cultural storytelling, including podcast narratives, sound-related material, and modern Greek heritage. |
| P9 | Professional tour guide or tour organizer | Spatial and itinerary-oriented exploration of sites, museums, regions, objects, and images. |
| P10 | Book-arts craftsperson | Byzantine and medieval bookmaking, binding structures, materials, and craft techniques. |
| P11 | Turkish scholar of folklore and ethnology | Shadow-theatre figures, social identities, and Ottoman or Turkish cultural representation. |
| P12 | Middle-school educator | Object-based examples for teaching ancient Greek history. |

Together, the personas test exhibition planning, archaeological research,
image retrieval, ethical documentation, classroom use, itinerary design,
cultural storytelling, craft inspiration, and sensitive heritage search.

## Representative information needs

The 55-query set included requests such as:

- identifying one representative object and image for each archaeological site
  represented in a museum;
- locating Greek burial sites in which obsidian tools were found;
- finding images of Mycenaean pottery from Crete;
- mapping places where human remains are stored or displayed;
- comparing typologies in representations of the Virgin Mary;
- retrieving artefacts with geometric structures that could inspire
  contemporary craft objects;
- assembling objects for a sound-rich story about twentieth-century Greece;
  and
- finding examples of Byzantine bookbindings with different sewing structures
  and materials.

These tasks deliberately mix direct lookup with grouping, comparison, spatial
reasoning, visual requirements, synthesis, and culturally sensitive questions.

## Systems and annotation protocol

For each query, the platform presented the persona, the query, and answers from
the four systems. The systems were shown through the same annotation interface
so that experts judged answer quality rather than interface differences.

Before rating answers, experts recorded how difficult the query would be to
answer manually in SearchCulture, estimated the required search time, and noted
the SearchCulture query or filters they used. They then evaluated each answer
using a fixed rubric covering:

| Dimension | What it examined |
|---|---|
| Hallucination | Whether the answer contained unsupported or invented claims. |
| Query relevance | How fully the answer addressed the stated information need. |
| Justification and evidence grounding | Whether claims were explained and supported by catalogue evidence. |
| Clarity and usefulness | Whether the answer was understandable and practically useful. |
| Bias and diversity | Whether the answer avoided bias and represented a sufficiently varied result set. |
| Completeness and representativeness | Whether important aspects and examples were missing. |
| Reasoning | Whether the logical steps needed to answer the query were present. |
| Comparison with manual search | Whether the answer was more or less useful than using SearchCulture directly. |
| Need for follow-up search | Whether the expert still needed to verify, extend, complete, or redo the search manually. |

For analysis, the 12 answer-level criteria were mapped to a common 0-3 scale,
with the need for further search reverse-coded. In the annotation interface,
manual-search usefulness used a five-point comparison scale and the
follow-up-search question used categorical response options.
<!-- 
## Main result

The full graph-guided CultureBot received the highest overall quality score
(2.34 out of 3), compared with ChatGPT (2.17), CultureBot without the knowledge
graph (2.00), and Gemini (1.82). Its clearest gains were in evidence grounding,
diversity, reasoning, and completeness. It was also the only system rated
significantly above parity with manual SearchCulture use, although experts still
considered follow-up search desirable for many questions.

These results should be read in light of the study's scope: it evaluates one
cultural heritage aggregator, all queries were written in English, and the
personas contain only a limited number of queries per role. The study measures
expert judgments of answers rather than longer-term outcomes for end users.

## Privacy and release scope

This page contains no participant names, contact details, raw annotations, or
credentials. Persona IDs describe fictionalized use cases and should not be
interpreted as identifiers for individual experts. The complete 55-query set
and anonymized annotation materials can be released separately once their
publication and licensing status is confirmed.

## How to cite

Please cite the CultureBot paper for the evaluation. Until its final
bibliographic record is available, link to this directory using a versioned
repository release or commit so that the referenced evaluation description
remains reproducible. -->
