
#%%
# Aanalys patients from TCGA-BRCA-counts file
import pandas as pd

STUDY = "TCGA-BRCA"
file_path = f"../data/omics/{STUDY}/{STUDY}.star_counts.tsv.gz"
df = pd.read_csv(file_path, sep='\t', compression='gzip', low_memory=False, nrows=3)

sample_type_codes = pd.read_csv("../data/omics/metadata/TCGA_sample_type_codes.csv")

""" 
Code,Sample Type,Description,Short Code
01,Primary Solid Tumor,Primary solid tumor,TP
02,Recurrent Solid Tumor,Recurrent solid tumor,TR
"""


def patient_code_parser(code):
    """Extract patient code from TCGA barcode.
    example code 'TCGA-C8-A274-01A'
    I need just sample type '01' 
    """
    parts = code.split('-')
    if len(parts) >= 3:
        sample_type = parts[3][:2]  # Get the first two characters of the fourth part
        return sample_type
    return None

# get all sample type from df.columns
sample_types = {col: patient_code_parser(col) for col in df.columns if col.startswith("TCGA")}
# show unique sample types and counts

for i in set(sample_types.values()):
    descriptinion = sample_type_codes[sample_type_codes['Code'] == int(i)]['Sample Type'].values

    print(f"Sample Type {i}: {list(sample_types.values()).count(i)} samples, type: {descriptinion}")


#%% --------------------------------------------------------------------------
from REHMED.src.data.omics_utils import read_data_type, PROBEMAP_PATHS

# print_df_heads()
DATA_TYPES = [
    "gene-level_ascat3",       # Copy Number (Gene Level)
    "allele_cnv_ascat3",       # Allele-specific Copy Number Segment
    #"methylation450",          # DNA Methylation - Illumina Human Methylation
    "methylation27",           # DNA Methylation - Illumina Human Methylation
    "somaticmutation_wxs",     # Somatic Mutation
    "mirna",                   # miRNA Expression
    "star_counts",             # Gene Expression (STAR - counts)
    "star_fpkm",               # Gene Expression (STAR - FPKM)
    "star_tpm",                # Gene Expression (STAR - TPM)
    "protein",                 # Protein Expression
    "clinical",                # Clinical Data
    "survival",                # Survival Data
]
df= read_data_type(DATA_TYPES[3])
print(df.columns)


# show csv of forst 3 rows and first 2 columns
print(df.iloc[:4, :3])
df.head(3).T.to_dict()
#%%
df.head(3).T.to_dict()
#%%
df.Ensembl_ID.nunique()
df["Ensembl_ID_nover"] = df["Ensembl_ID"].str.split('.').str[0]
df["Ensembl_ID_nover"].nunique()

#%%
gene_mappings = pd.read_csv(PROBEMAP_PATHS[0], sep='\t')
gene_mappings.gene.nunique()

# count only gene that ends with "P"
number_of_pseudo = gene_mappings[gene_mappings['gene'].str.endswith('P')].gene.nunique()   
print(f"Number of pseudo genes: {number_of_pseudo}")


# count only gene that starts with "MIR"
number_of_mirna = gene_mappings[gene_mappings['gene'].str.startswith('MIR')].gene.nunique()   
print(f"Number of miRNA genes: {number_of_mirna}")

# count only gene that contains with "."
number_of_genes_with_dot = gene_mappings[gene_mappings['gene'].str.contains('\.')].gene.nunique()
print(f"Number of genes containing '.': {number_of_genes_with_dot}")


gene_mappings["gene_nodot"] = gene_mappings["gene"].str.split('.').str[0]
genenodot_num = gene_mappings["gene_nodot"].nunique()
print(f"Number of unique genes without dot: {genenodot_num}")
#%%
