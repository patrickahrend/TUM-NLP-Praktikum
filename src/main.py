import pandas as pd

def process_matching_data(excel_file, sheet_name, output_file):
    matching_data = excel_file.parse(sheet_name)
    
    df_labels = matching_data[['requirement_text', '0 = not relevant; 1 = business compliance relevance; 2 = (customer) informative relevance']].copy()
    df_labels = df_labels.rename(columns={
        '0 = not relevant; 1 = business compliance relevance; 2 = (customer) informative relevance': 'label',
        'requirement_text': 'Text'
    })
    

    df_labels['label'] = df_labels['label'].replace(2, 1)
    

    df_labels.to_csv(output_file, index=False)
    
    print(f'Processed {sheet_name}:')
    print(df_labels['label'].value_counts())


australinen_excel = pd.ExcelFile("../data/Australia_Use_Cases.xlsx")


process_matching_data(australinen_excel, "1_matching_reordered", "../data/1_Matching_Labels.csv")
process_matching_data(australinen_excel, "2_matching_reordered", "../data/2_Matching_Labels.csv")
