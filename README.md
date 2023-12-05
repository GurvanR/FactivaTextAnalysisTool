# FactivaTextAnalysisTool

Python text-analysis pseudo library specifically made to manipulate files from the Factiva database.

- First download rtf files from the Factiva database: [Factiva Database](https://www.dowjones.com/professional/factiva/). You can freely access it if you are a student in an ENS, for instance, via this [link](https://bib.ens.psl.eu/ulm-lsh-jourdan-shs/collections/liste-des-bases-de-donnees-et-bouquets-de-revues-et-de-books-en). Search for 'factiva' in the provided link.
  - Note that you can download only 100 articles at a time from this database, making the collecting process a bit lengthy.
  - Ensure you download rtf files in the correct format, which is the "Article Format." Test the analysis with a batch of 100 articles before downloading more articles.

- Once you have your rtf files, download the repository and run the Tuto notebook to explore the functionalities available for factiva files.

- See requirements.txt to create the corresponding environment necessary for running the code.

- Enjoy exploring the functionalities and feel free to reach out if you have any remarks or questions.

PS: There might be trouble displaying Altair stuff (for TF-IDF and topic modeling, for instance). The `altair_viewer.show()` from the `altair_viewer` library may not work. The only workaround found is to let the notebook display it by simply running the cell with the variable.

