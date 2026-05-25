#%%

# set working dir
import os

def set_working_directory():
    print(f"Current working directory: {os.getcwd()}")
    if os.getcwd().endswith("scripts"):
        print("Changing working directory to ..\\data\\pkt\\builds\\v3.0.2")
        os.chdir("..\\data\\pkt\\builds\\v3.0.2")
        print(f"Current working directory: {os.getcwd()}")

def print_file_contents(file_path=os.getcwd(), num_lines=10):
    """Print the files contained in the working directory sorted for file size and extension.
    """
    files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    file_info = []
    for f in files:
        full_path = os.path.join(file_path, f)
        size = os.path.getsize(full_path)
        ext = os.path.splitext(f)[1]
        file_info.append((f, size, ext))
    
    # Sort by size (descending) and then by extension
    file_info.sort(key=lambda x: (-x[1], x[2]))
    
    print(f"Files in directory: {file_path}\n")
    for f, size, ext in file_info:
        print(f"File: {f} -  Size: {size / (1024*1024):.2f} MB -  Extension: {ext}\n")
        # Print first num_lines of the file
        # try:

        #     with open(os.path.join(file_path, f), 'r', encoding='utf-8') as file:
        #         print(f"  First {num_lines} lines:")
        #         for i in range(num_lines):
        #             line = file.readline()
        #             if not line:
        #                 break
        #             print(f"    {line.strip()}")
        # except Exception as e:
        #     print(f"  Could not read file: {e}")
        print("-" * 40)


import pandas as pd
import tarfile

def read_tar_rdf(file_path):
    print(f"Reading file ID {file_path}")    
    with tarfile.open(file_path, 'r:gz') as tar:
        members = tar.getmembers()
        # Filtra i file il cui nome non inizia con "."
        valid_members = [m for m in members if not m.name.startswith('.')]
        # Estrarre il primo file valido
        extracted_file = tar.extractfile(valid_members[0])
        df = pd.read_csv(extracted_file, sep=' ', header=None, names=['subject', 'predicate', 'object', '.'])
        return df[['subject', 'predicate', 'object']]

def read_tar_tsv(file_path): 
    print(f"Reading file ID {file_path}")    
    with tarfile.open(file_path, 'r:gz') as tar:
        members = tar.getmembers()
        # Filtra i file il cui nome non inizia con "."
        valid_members = [m for m in members if not m.name.startswith('.')]
        # Estrarre il primo file valido
        extracted_file = tar.extractfile(valid_members[0])
        df = pd.read_csv(extracted_file, sep='\t')
        return df


# def read_tar_rdf(file_path):
#     print(f"Reading file ID  {file_path}")    
#     with tarfile.open(file_path, 'r:gz') as tar:
#         members = tar.getmembers()
#         extracted_file = tar.extractfile(members[0])
#         df = pd.read_csv(extracted_file, sep=' ', header=None, names=['subject', 'predicate', 'object', '.'])
#         return df[['subject', 'predicate', 'object']]

# def read_tar_tsv(file_path): 
#     print(f"Reading file ID  {file_path}")    
#     with tarfile.open(file_path, 'r:gz') as tar:
#         members = tar.getmembers()
#         extracted_file = tar.extractfile(members[0])
#         df = pd.read_csv(extracted_file, sep='\t')
#         return df

# =====================================================================
import requests

def download_build_files():
    """ Download and build PKT files from Zenodo.
    al posto di file_root usare "PKT" nel nome finale del file
    """
    # Download and build PKT files
    # https://zenodo.org/records/10056202

    file_root = "PheKnowLator_v3.0.2_full_instance_inverseRelations_OWLNETS_INSTANCE_purified"
    files = [
        f"{file_root}.nt.tar.gz",
        # f"{file_root}_decoding_dict.pkl.tar.gz",
        # f"{file_root}_Triples_Identifiers.txt.tar.gz",
        f"{file_root}_NodeLabels.txt.tar.gz",
        # f"{file_root}_Triples_Integers.txt.tar.gz",
        # f"{file_root}_Triples_Integer_Identifier_Map.json.tar.gz", 
        f"{file_root}_NetworkxMultiDiGraph.gpickle.tar.gz",
        "Master_Edge_List_Dict.json.tar.gz",
        "ontology_source_list.txt.zip",
        "edge_source_list.txt.zip",
        "downloaded_build_metadata.txt.zip",
        "node_metadata_dict.pkl.tar.gz",
    ]

    root_path = "../data/pkt/builds/v3.0.2/"
    # create directory if not exists
    os.makedirs(root_path, exist_ok=True)
    
    for file in files:
        url = f"https://zenodo.org/records/10056202/files/{file}?download=1"
        output_path = os.path.join(root_path, file.replace(file_root, "PKT"))
        print(f"Downloading {url} to {output_path}")
        # os.system(f"wget -O {output_path} '{url}'")
        
        # Scarica con requests
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Controlla errori

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded {output_path}")


#%%
# =====================================================================
if __name__ == "__main__":
    # set_working_directory()

    # print_file_contents()

    download_build_files()
    print("")
# %%
