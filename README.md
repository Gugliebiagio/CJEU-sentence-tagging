# Court-of-Justice-of-the-European-Union-CJEU-sentence-tagging

Project of 6 CFU in NLP.
Given a dataset of CJEU we have that the tagging involves two types of argumentative elements: premises and conclusions. Each premise can be factual, legal, or both (attribute T), and each legal-type premise is associated with one or more argumentative schemes (attribute S). The tasks, therefore, include argument classification (premises/conclusions), type classification (factual/legal).
The classes are highly unbalanced, especially in argument and schema classification. Therefore, the proposal is to try some data augmentation techniques and/or data cartography with curriculum learning.
The work is divided into three tasks:
- Classify argumentative sentences as premises or conclusions (AC).
- Distinguish between legal and factual premises (TC).
- Identify argumentative patterns (SC).

Data Augmentation files:
- Augmentation: creates the Augmented Dataset for each of the three tasks
- Classification: reports the result for each embedding and classificator with the augmented dataset on the three tasks.

Data Cartography with Curriculum Learning and Data Cartography + Data Augmentation + Curriculum Learning:
- Cartography_AC: applies cartography and curriculum learning techniques for the AC task, as well as the three techiniques together
- Cartography_TC: applies cartography and curriculum learning techniques for the TC task, as well as the three techiniques together
- Cartography_SC: applies cartography and curriculum learning techniques for the SC task, as well as the three techiniques together
