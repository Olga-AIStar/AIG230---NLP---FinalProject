Project Progress Report: Intelligent Complaint Onboarding System (ICOS)

The CFPB https://www.consumerfinance.gov/about-us/the-bureau/

 the Consumer Financial Protection Bureau, a U.S. government agency dedicated to making sure you are treated fairly by banks, lenders, and other financial institutions.

Data dictionary : https://www.consumerfinance.gov/complaint/data-use/
Data masking: https://files.consumerfinance.gov/f/documents/cfpb_narrative-scrubbing-standard_2023-05.pdf



1. Introduction 
The goal of this project is to analyze, classify and summarize data  from the Consumer Financial Protection Bureau (CFPB), a U.S. government agency dedicated to ensuring fair treatment by financial institutions. This report focuses on the public Consumer Complaint Database, which serves as a transparent record of consumer-company interactions in the financial sector. 

2. Dataset Overview & Scope
            The initial data exploration involved a massive raw dataset. 
            Original File Size: 7.52 GB
            Total Records: 13,455,357
            Total Columns: 18
            Data Masking: Following the CFPB's Narrative Scrubbing Standard, personal identifiers (names, dates, account numbers) are masked using "XXXX" variations to protect privacy. 

3. Explanatory Data Analysis (EDA) and Missing Values
    We filtered the dataset to only include complaints where the consumer explicitly provided consent to share their narrative, reducing the working data to 3.7 million records. A missing values scan revealed that the core "Consumer complaint narrative" column is nearly perfect, with only 0.04% missing data. Additionally, while the "Company public response" is missing in nearly half of the records, the "Company response to consumer" metric is almost 100% complete. These metrics confirm that we have a highly reliable text feature for our NLP model.
    Dataset contains 3,704,714 rows and 18 columns.

            Missing Values Report:

                                        tot_missing  % of overall
            Consumer disputed?                3540683     95.572371
            Tags                              3355183     90.565237
            Company public response           1708102     46.106177
            Sub-issue                          320464      8.650168
            Sub-product                         52214      1.409394
            State                               12984      0.350472
            Consumer complaint narrative         1658      0.044754
            Company response to consumer            9      0.000243
            Date received                           0      0.000000
            Submitted via                           0      0.000000
            Timely response?                        0      0.000000
            Date sent to company                    0      0.000000
            ZIP code                                0      0.000000
            Consumer consent provided?              0      0.000000
            Product                                 0      0.000000
            Company                                 0      0.000000
            Issue                                   0      0.000000
            Complaint ID                            0      0.000000




4. The Challenge of Boilerplate Templates column ‘Consumer complaint narrative’
        Our initial analysis identified a high volume of "form letter" complaints dominating the dataset. The most frequent record appeared as an identical entry over 27,000 times. Furthermore, the top ten most common narratives are simply copy-pasted legal templates citing "15 U.S. Code 1681" from the Fair Credit Reporting Act.

Column [Consumer complaint narrative] has 2,472,393 unique values, max/min content lenghts 35,984/4 ; avg lenght 1,195

        Printing top 10 records out of 2,472,393
        -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Value                                                                                                                                                  | Total Records        | Percentage     
        -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        In accordance with the Fair Credit Reporting act. The List of accounts below has violated my federally protected consumer rights to privacy and        | 27,389               | 0.74%
        confidentiality under 15 USC 1681.  15 U.S.C 1681 section 602 A. States I have the right to privacy.  15 U.S.C 1681 Section 604 A Section 2 : It            also  |                      |                
        states a consumer reporting agency can not furnish a account without my written instructions 15 U.S.C 1681c. ( a ) ( 5 ) Section States : no consumer  |                      |                
        reporting agency may make any consumer report containing any of the following items of information Any other adverse item of information, other than   |                      |                
        records of convictions of crimes which antedates the report by more than seven years.  15 U.S.C. 1681s-2 ( A ) ( 1 ) A person shall not furnish any    |                      |                
        information relating to a consumer to any consumer reporting agency if the person knows or has reasonable cause to believe that the information is     |                      |                
        inaccurate.                                                                                                                                            |                      |                
        ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------- | ---------------


5. Removing Duplicated Complaints.
        Retaining these identical templates would lead to significant data leakage, causing the model to simply memorize legal formats rather than learn actual language patterns. This over-representation of legalese would force the model's vocabulary to heavily favor specific legal citations over natural human descriptions. Ultimately, failing to remove these duplicates would artificially inflate the "Credit Reporting" category, resulting in a heavily biased and inaccurate model.

6. Addressing Data Imbalance and Label Overlap
        The Product classification target presented two main challenges: overlapping historical labels and extreme class imbalance, with credit reporting products dominating 66% of the dataset.To resolve this and ensure unbiased model training, we implemented the following pipeline:
            Column [Product] has 21 unique values, max/min content lenghts 76/8 ; avg lenght 26

            Printing top 10 records out of 21
            ---------------------------------------------------------------------------------------------------------------------
            Value                                                                        | Total Records        | Percentage     
            ---------------------------------------------------------------------------------------------------------------------
            Credit reporting or other personal consumer reports                          | 1,650,218            | 44.56%
            Credit reporting, credit repair services, or other personal consumer reports | 807,271              | 21.80%
            Debt collection                                                              | 401,823              | 10.85%
            Checking or savings account                                                  | 166,563              | 4.50%
            Mortgage                                                                     | 138,984              | 3.75%
            Money transfer, virtual currency, or money service                           | 111,086              | 3.00%
            Credit card or prepaid card                                                  | 108,666              | 2.93%
            Credit card                                                                  | 107,237              | 2.90%
            Student loan                                                                 | 58,770               | 1.59%
            Vehicle loan or lease                                                        | 47,195               | 1.27%
            Credit reporting                                                             | 31,587               | 0.85%

        **Label Consolidation:** We grouped the 21 original, overlapping categories into 8 high-level classifications within a new Product_Group column.
        Stratified Downsampling: We extracted a representative subset of 100,000 records. This allows for efficient local training while maintaining the original proportions of the 8 major groups.
        Hybrid Resampling: To perfectly balance the classes within this subset, we undersampled dominant majority groups (like Credit Reporting) and oversampled minority groups (like Virtual Currency). This ensures the model receives enough examples to reliably recognize every category.
7. Feature Engineering
    Introducing Frustration and Regulatory Indices to capture the emotional intensity and legal complexity of each narrative.

8. Target Variable Encoding
    To prepare the dataset for machine learning classification, the textual labels had to be converted into a numerical format. 

9. Text Normalization & Tokenization 
A custom function was applied to convert all narratives to lowercase and strip out extraneous punctuation using regular expressions. The text was then tokenized into individual words and processed through a WordNet Lemmatizer to reduce each word to its root morphological form. 

10. Data Splitting: 80%; 10%;10% 
Using predefined configuration ratios, the data was split into a 32,000-record training set, a 4,000-record validation set, and a 4,000-record test set.

11. Vocabulary Generation & Encoding 
 A formal numerical vocabulary was constructed exclusively from the training set to ensure the model does not artificially gain knowledge from the testing data.





