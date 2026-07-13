# MetaQA Results (reference values)

These are the numbers reported as Table 17 (Appendix L) of the paper. They are
committed here as a reference target for reproduction; the pipeline that
produces them is documented in [`data/METAQA.md`](data/METAQA.md).

Model: ReaRev backbone (GNN-RAG), trained per hop from scratch (sbert encoder,
10 epochs). Evaluation: full test sets, 14,872 questions (2-hop) and 14,274
(3-hop). Perturbations: all seven types, generated with the same Llama-3.3-70B
prompts and validity gates as the Freebase experiments.

## Clean baselines

| Split | Hit@1 | F1 | EM |
|---|---|---|---|
| MetaQA-2hop | 99.74 | 98.66 | 99.97 |
| MetaQA-3hop | 95.88 | 75.28 | 98.10 |

## Fixed-subgraph condition (question perturbed, subgraph held fixed)

Hit@1 (%), with the drop from the corresponding clean baseline in percentage
points.

| Code | Attack | 3-hop Hit@1 | 3-hop delta | 2-hop Hit@1 | 2-hop delta |
|---|---|---|---|---|---|
| p1 | ES (entity alias) | 88.97 | 6.91 | 96.61 | 3.13 |
| p2 | S1 (relation paraphrase) | **82.79** | **13.09** | **86.77** | **12.97** |
| p3 | S2 (voice flip) | 92.05 | 3.83 | 98.13 | 1.61 |
| p4 | CR (compositional restructuring) | 88.97 | 6.91 | 99.63 | 0.11 |
| p5 | S3 (entity insertion) | 95.03 | 0.85 | 99.35 | 0.39 |
| p6 | S4 (mixed noise) | 88.76 | 7.12 | 95.81 | 3.93 |
| p7 | RS (relation synonym) | 90.94 | 4.94 | 97.40 | 2.34 |

## Findings

1. **No collapse under any perturbation type.** The worst attack (S1, relation
   paraphrase) costs 13.09 pp on 3-hop. By contrast, the same CR attack drives
   GNN-RAG from 52.9% to 0.68% EM on CWQ, a 52.22 pp collapse. Since the
   generator, the validity filters, and the GNN architecture are held fixed
   across the two settings, the difference is attributable to the retrieval
   stage: MetaQA expands a seed neighbourhood rather than computing a
   PPR-weighted subgraph, so a restructured question cannot misdirect a PPR walk
   that is never computed.

2. **Retrieval invariance is measured, not assumed.** Re-running the entire
   pipeline (entity linking, subgraph construction, GNN inference) on the
   perturbed questions changes Hit@1 by at most 0.6 pp under CR on either hop,
   relative to the same pipeline on clean text.

3. **The EL-conflation caveat reproduces cross-KB.** S3 (adversarial entity
   insertion) degrades an automatic longest-match linker by 22.7 pp (3-hop) and
   18.6 pp (2-hop), because the injected distractor becomes the longest surface
   match, and this propagates to an 8.3 pp end-to-end drop. Under the gold seeds
   that published systems consume, the same attack costs 0.4 pp. Its apparent
   harmlessness in the table above is a consequence of the seeding convention,
   not evidence that distractor injection is benign for a deployed linker.

## Caveats

- Absolute accuracies in the end-to-end condition are bound to the reference
  linker and retriever (clean gold-seeded Hit@1 is 30.69% on 3-hop, well below
  the 95.88% obtained with the subgraphs shipped with the dataset). Only
  within-condition deltas should be compared.
- ES is excluded from the end-to-end comparison because it changes the gold
  answer by design, so clean and perturbed runs do not measure the same question.