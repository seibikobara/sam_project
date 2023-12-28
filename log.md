# NLP project

# log
## 12.28.23

- keep the current analysis
- disproportional analysis, which can account for COVID-19 pandemic or any baseline symptom distribution, may not be feasible due to a small sample size.
- authorship? we want to have three authors are equally contributed.


## Methods
- Dataset
  - Breast cancer twitter


### Objectives
1. Classification of breast cancer posts
   1. using Dr. Abeed's annotation dictionary
      1. breast cancer-related post or not.
      2. train using the file and expand to the whole data to identify breast cancer-related posts
2. Lexicon analysis
   3. Medication
      1. inexact match, spell check
   4. Adverse side effect
      1. manual annotation to discover. 
      2. lexicon identification
      3. rule-based 
3. Sentiment analysis or association analysis or clustering?


### How to evaluate for each objective
1. Compare with gold standard
   1. This can be achieved using SOMN441?
2. Manually?
3. Manually? 


### Date use agreement
Nov 2 sent to Abeed and Mengyu


### Tasks
- Intro 500 words by Tuesday 7th 
  - brief background
  - what are the unclear research questions
  - our objectives
  - hypotheses
  - resources used for these objectives 
   1. clasification: annotated dataset
   2. drug: FDA etc.
   3. sentiment: 
  - ways for evaluation of our models and validation
   1. classificaiton: test set in the abeed's dataset
   2. ?
   3. ?
   

- By Monday
  - Alireza and Masoud would run an initial try for classification
  - SK will find useful database for drug names and side effect
  - Come up with the 3rd objective.



## References
### Extracting Drug Names and Associated Attributes From Discharge Summaries: Text Mining Study
- [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8135022/]
- Drug: The chemical name of a drug or the advertised brand name under which a drug is sold (eg, aspirin)
- Dosage: The amount of medicine that the patient takes or should take (eg, 2 tablets, 5 mL)
- Strength: The amount of drug in a given dosage (eg, 200 mg)
- Frequency: The rate at which medication was taken or is repeated over a particular period (eg, daily, every 4 hours)
- Duration: The period of continuous medication taking (eg, pro re nata, for 5 days)
- Route: The path by which medication is taken into the body or the location at which it is applied (eg, topical, per os)
- Form: The form in which a medication is marketed for use (eg, tablet)
- Reason: The reason for medication administration (eg, for pain)


## How can natural language processing help model informed drug development?: a review 
- [https://academic.oup.com/jamiaopen/article/5/2/ooac043/6605908]
- NLP models for MIDD (model informed drug development)
- BioBert, ClinicalBERT, and other around 10.


## Information Extraction From FDA Drug Labeling to Enhance Product-Specific Guidance Assessment Using Natural Language Processing
- [https://www.frontiersin.org/articles/10.3389/frma.2021.670006/full]
- Drugs@FDA
  - [https://www.fda.gov/drugs/development-approval-process-drugs/drug-approvals-and-databases]
  - Drugs@FDA is a publicly available resource, which includes the majority of drug labeling, approval letters, reviews, and other information for FDA-approved drug products for human use provided by the FDA. It contains prescription brand-name drug products, over-the-counter brand-name drug products, generic drug products, and therapeutic biological products.
- DailyMed
  - DailyMed is a free drug information resource provided by the United States. National Library of Medicine (NLM) that consists of digitized versions of drug labeling as submitted to the FDA. It is the official provider of the FDA labeling information (package inserts). The documents published use the Health Level Seven (HL7) version 3 Structured Product Labeling (SPL) standard, which specifies various drug label sections (Schadow 2005, 7). It uses Logical Observation Identifiers Names and Codes (LOINC) to link sections and subsections of human prescription drug and biological product labeling.
- DrugBank
  - DrugBank is a richly annotated resource that combines detailed drug data with comprehensive drug target and drug action information provided by the University of Alberta and the Metabolomics Innovation Center (Wishart et al., 2008). It contains FDA-approved small molecule and biologics drugs with extensive food-drug and drug-drug interactions as well as ADMET (absorption, distribution, metabolism, excretion, and toxicity) information (Knox et al., 2011).