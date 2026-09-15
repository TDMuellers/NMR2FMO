# load needed packages
import os
import numpy as np
import pandas as pd
import re
from scipy.stats import mode, skew, kurtosis
from rdkit import Chem
from rdkit.Chem import Fragments, Lipinski, rdDistGeom, rdMolDescriptors, SanitizeMol, rdchem, Descriptors

#### Functions for data extraction ####

def load_qm9(xyz_folder):
    """
    Reads all .xyz files in the given folder and returns a pandas DataFrame
    with named columns for each value, including the filename.
    Data from: https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904
    """
    columns = [
        "filename", "num_atom", "dataset", "index", "rotational_a_ghz", "rotational_b_ghz", "rotational_c_ghz",
        "dipole_debye", "isotropic_pol_bohr3", "homo_har", "lumo_har", "gap_har", "elec_spat_bohr2",
        "zpve_har", "u0_har", "u_har", "h_har", "g_har", "cv_har", "ir_freqs", "smiles", "smiles_relax",
        "inchi", "inchi_relax"
    ]
    df_storage = []
    for filename in os.listdir(xyz_folder):
        if filename.endswith('.xyz'): # there should only be .xyz in folder, but just in case
            file_path = os.path.join(xyz_folder, filename)
            with open(file_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            num_atoms = lines[0]
            data_line = lines[1]
            # Find atom block start
            atom_start = None
            for i, line in enumerate(lines):
                if len(line) > 0 and line[0].isalpha() and (line[1] == '\t' or line[1] == ' '):
                    atom_start = i
                    break
            # Find atom block end
            atom_end = atom_start
            while atom_end < len(lines) and lines[atom_end][0].isalpha():
                atom_end += 1
            # IR frequencies
            # starts after atom block (not extracted)
            ir_line = lines[atom_end]
            ir_freqs = np.array([float(x) for x in ir_line.split()])
            # String values
            # SMILES and InChI
            string_values = []
            for line in lines[atom_end+1:]:
                if line:
                    string_values.extend(line.split('\t'))
            # Compose row
            values = [filename] + [num_atoms] + data_line.split() + [ir_freqs] + string_values[:4]
            df_storage.append(values)
    df = pd.DataFrame(df_storage, columns=columns)
    return df

def load_qm9_nmr(filename):
    """
    Load NMR data from QM9NMR format file into a pandas DataFrame.
    This is based on information at https://moldis-group.github.io/qm9nmr/
    """

    # determine columns based on data structure
    columns = [
        "num_atom", "filename", "atoms", "nmr_gas", "nmr_CCl4", "nmr_THF",
        "nmr_Acetone", "nmr_Methanol", "nmr_DMSO"
    ]
    data = []
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    i = 0
    while i < len(lines):
        # get descriptive data for each entry
        num_atoms = lines[i]
        qm9_id = lines[i+1]
        atom_lines = []
        i += 2 # skips first two lines
        # as long as there are more lines and not a digit (the next entry)
        while i < len(lines) and not lines[i].isdigit(): 
            atom_lines.append(lines[i])
            i += 1
        # Parse atom lines
        for line in atom_lines:
            split_atoms = [line.split() for line in atom_lines]
            # First array: atom letters
            letters = [row[0] for row in split_atoms]
            # Remaining arrays: columns of values
            values = [[float(row[i]) for row in split_atoms] for i in range(1, len(split_atoms[0]))]
            # Result: [letters, values...]
            result = [letters] + values
        values = [num_atoms] + [qm9_id] + result
        data.append(values)
    df = pd.DataFrame(data, columns=columns)
    return df

def load_sdf_to_df(sdf_path):
    """
    Loads a large SD file, preserving all fields, and returns a pandas DataFrame.
    Each row is a molecule, columns are SD fields (including structure as Mol object).
    """
    suppl = Chem.SDMolSupplier(sdf_path)
    records = []
    for mol in suppl:
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        props['rdkit_mol'] = mol  # Optionally store the molecule object
        records.append(props)
    df = pd.DataFrame(records)
    return df

#### Featurization functions ####

def atom_select(df, atom_col, signal_col, atom_type):
    """
    For each row, selects only the signals for the specified atom type.
    Returns a new DataFrame with one row per molecule and a vector of signals for that atom.
    atom_type = user input string, e.g. "C" or "H"
    atom_col = name of column with atom types (list of strings)
    """
    def filter_signals(atoms, signals):
        return [s for a, s in zip(atoms, signals) if a == atom_type]

    # Apply filtering to each row and add as a new column
    df[f"{atom_type}_signals"] = df.apply(lambda row: filter_signals(row[atom_col], row[signal_col]), axis=1)
    return df

def get_summary_stats(df, array_col):
    """
    For each row, computes:
    length, min, max,
    arithmetic mean,
    median, mode,
    std, variance,
    skewness, and kurtosis.
    Returns a DataFrame with these statistics appended to the input.
    """
    
    # Prepare lists to hold each statistic
    lengths, mins, maxs, arith_means, medians, modes, vars, stds, skews, kurtoses = [], [], [], [], [], [], [], [], [], []
    for idx, row in df.iterrows():
        arr = np.array(row[array_col])
        lengths.append(len(arr))
        mins.append(np.min(arr))
        maxs.append(np.max(arr))
        arith_means.append(np.mean(arr)) # numpy standard arithmetic mean
        medians.append(np.median(arr))
        modes.append(mode(arr)[0]) # scipy mode
        vars.append(np.var(arr, ddof=0))
        stds.append(np.std(arr, ddof=0))
        skews.append(skew(arr, nan_policy='raise'))
        kurtoses.append(kurtosis(arr, nan_policy='raise'))

    # Add new columns to the input DataFrame
    df[f"{array_col}_length"] = lengths
    df[f"{array_col}_min"] = mins
    df[f"{array_col}_max"] = maxs
    df[f"{array_col}_mean"] = arith_means
    df[f"{array_col}_median"] = medians
    df[f"{array_col}_mode"] = modes
    df[f"{array_col}_var"] = vars
    df[f"{array_col}_std"] = stds
    df[f"{array_col}_skewness"] = skews
    df[f"{array_col}_kurtosis"] = kurtoses
    return df

def get_all_fgs(df, smiles_col='smiles'):
    """
    Return a DataFrame with functional group columns (e.g., Alcohol, Amine, etc.)
    for each molecule in the input DataFrame, using CAS numbers to resolve SMILES.
    refer to https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html
    """
    # discover all fragment function names from rdkit.Chem.Fragments
    frag_names = sorted([name for name in dir(Fragments) if name.startswith('fr_')])

    # Prepare a dictionary for each fragment column
    fg_cols = {fname: [] for fname in frag_names}

    for idx, row in df.iterrows():
        smiles = row.get(smiles_col) if smiles_col in row else None     
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                for fname in frag_names:
                    try:
                        func = getattr(Fragments, fname)
                        val = func(mol)
                        if val is None:
                            fg_cols[fname].append(0)
                        else:
                            try:
                                fg_cols[fname].append(int(val))
                            except Exception:
                                fg_cols[fname].append(val)
                    except Exception:
                        fg_cols[fname].append(np.nan)
            else:
                for fname in frag_names:
                    fg_cols[fname].append(np.nan)
        else:
            for fname in frag_names:
                fg_cols[fname].append(np.nan)

    # Add new columns to the input DataFrame
    for fname in frag_names:
        df[fname] = fg_cols[fname]
    return df

def get_lipinski_descriptors(df, smiles_col='smiles'):
    """
    Compute all callable descriptors from rdkit.Chem.Lipinski for each SMILES in df.
    Returns a DataFrame with a 'smiles' column and one column per Lipinski function.
    """

    # discover Lipinski functions (public callables)
    lip_names = sorted([name for name in dir(Lipinski)
                        if not name.startswith('_') and callable(getattr(Lipinski, name))])

    # Prepare a dictionary for each Lipinski descriptor column
    lip_cols = {fname: [] for fname in lip_names}

    for idx, row in df.iterrows():
        smiles = row.get(smiles_col)
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                for fname in lip_names:
                    try:
                        func = getattr(Lipinski, fname)
                        val = func(mol)
                        if val is None:
                            lip_cols[fname].append(np.nan)
                        else:
                            if isinstance(val, bool):
                                lip_cols[fname].append(int(val))
                            else:
                                try:
                                    lip_cols[fname].append(int(val))
                                except Exception:
                                    try:
                                        lip_cols[fname].append(float(val))
                                    except Exception:
                                        lip_cols[fname].append(val)
                    except Exception:
                        lip_cols[fname].append(np.nan)
            else:
                for fname in lip_names:
                    lip_cols[fname].append(np.nan)
        else:
            for fname in lip_names:
                lip_cols[fname].append(np.nan)

    # Add new columns to the input DataFrame
    for fname in lip_names:
        df[fname] = lip_cols[fname]
    return df

def is_inchikey(s):
    # InChIKeys are 27 characters, formatted as "AAAAAA-BBBBBB-CCCCCC"
    return isinstance(s, str) and bool(re.match(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$", s))

def get_rdkit_constitutional(mol): 
    """
    get mol formula, number of heavy atoms, number of carbons, number of hydrogens
    this operates on existing "mol" column provided in original NMRSshiftDB
    """
    if mol is None:
        return pd.Series({'molecular_formula': None,
                          'molecular_weight': None, 
                          'num_heavy_atoms': None, 
                          'num_carbons': None,
                          'num_hydrogens': None})
    try:
        formula = rdMolDescriptors.CalcMolFormula(mol)
        mol_weight = Descriptors.ExactMolWt(mol)
        num_heavy = mol.GetNumHeavyAtoms()
        num_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
        # Extract integer after "H" in formula, or NaN if not present
        match = re.search(r'H(\d+)', formula) # get H and integer following it
        if match:
            num_hydrogens = int(match.group(1))
        else:
            num_hydrogens = np.nan
        return pd.Series({'molecular_formula': formula,
                          'molecular_weight': mol_weight,
                          'num_heavy_atoms': num_heavy,
                          'num_carbons': num_carbons,
                          'num_hydrogens': num_hydrogens})
    except Exception:
        return pd.Series({'molecular_formula': None,
                          'molecular_weight': None, 
                          'num_heavy_atoms': None, 
                          'num_carbons': None,
                          'num_hydrogens': None})
    
def get_index(colname):
    # Helper to get the integer index from column name
    m = re.search(r'(\d+)$', colname)
    return int(m.group(1)) if m else None

def extract_index_value_pairs(cell):
    """
    get indices and values for nmr conditions
    Given a cell like '0:303 1:Unreported', return two lists:
    - index_check: [0, 1]
    - value: ['303', 'Unreported']
    """
    if pd.isna(cell):
        return [], []
    # this is a crucial regex
    # (\d+): Matches one or more digits (the index before the colon).
    #:: Matches the literal colon separating index and value.
    # (.*?): Non-greedy match for any characters (the value after the colon).
    # (?=\s\d+:|$): Lookahead for either a space followed by digits and a colon 
    # (the start of the next index-value pair), or the end of the string.
    matches = re.findall(r'(\d+):(.*?)(?=\s\d+:|$)', cell)

    indices = [int(m[0]) for m in matches]
    values = [m[1].strip() for m in matches]

    return indices, values

def extract_first_floats(df, col):
    """
    Take nmrshiftdb2 style data and replace it with an array of chemical shifts
    """
    def parse_cell(cell):
        if pd.isna(cell):
            return []
        groups = str(cell).split('|')
        floats = []
        for group in groups:
            parts = group.split(';')
            try:
                floats.append(float(parts[0]))
            except (ValueError, IndexError):
                continue
        return floats
    df[col] = df[col].apply(parse_cell)
    return df

def signals_match(row, signal_col, count_col):
    # this function checks if the number of signals matches the count of relevant atoms
    val = row[signal_col]
    if val is None or (hasattr(val, 'size') and val.size == 0):
        return False
    elif isinstance(val, (list, tuple)):
        signal_count = len(val)
    return row[count_col] == signal_count

def deduplicate_spectra(df, solvent_col='Solvent', 
                        preferred_solvent='Chloroform-D1 (CDCl3)',
                        id_col='INChI key',
                        second_preferred_solvent=None):
    # Sort so preferred solvents come first within each group
    df_sorted = df.copy()
    # Assign priority: 2 for first preferred, 1 for second preferred, 0 otherwise
    df_sorted['solvent_priority'] = 0
    df_sorted.loc[df_sorted[solvent_col] == preferred_solvent, 'solvent_priority'] = 2
    if second_preferred_solvent is not None:
        df_sorted.loc[df_sorted[solvent_col] == second_preferred_solvent, 'solvent_priority'] = 1
    df_sorted = df_sorted.sort_values(['solvent_priority'], ascending=False)
    df_sorted[id_col] = df_sorted[id_col].apply(lambda x: str(x) if isinstance(x, list) else x)
    # For each id_col, keep the first row (which will be preferred solvent if present)
    deduped = df_sorted.drop_duplicates(subset=[id_col], keep='first')
    # Drop helper column
    #deduped = deduped.drop(columns=['solvent_priority'])
    return deduped

def is_valid_mol(mol):
    try:
        SanitizeMol(mol)
        return True
    except rdchem.AtomValenceException:
        return False
    except Exception:
        return False

def process_directory(directory):
    """Process all files in a directory and compile them into a DataFrame.
    Note that we do this for all files, not only .log
    """
    all_data = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        data = get_data_from_log(file_path)
        if data is not None:
            all_data.append(data)

    df = pd.DataFrame(all_data)
    return df

def get_data_from_log(file_path):
    """Extract required molecular properties from a Gaussian log file."""
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Ensure we only consider content after "Optimization completed."
    optimization_completed_index = content.find("Optimization completed.")
    if optimization_completed_index == -1:
        return None
    
    # start after the found phrase
    content_after_optimization = content[optimization_completed_index + len("Optimization completed."):]
    
    # Updated for capturing the specific floating point values from the final blocks
    ehomo_blocks = re.findall(r'Alpha  occ\. eigenvalues -- [^0-9.-]*((?:-?\d+\.\d+[\s]*)+)', content_after_optimization)
    elumo_blocks = re.findall(r'Alpha virt\. eigenvalues -- [^0-9.-]*((?:-?\d+\.\d+[\s]*)+)', content_after_optimization)
    
    # Extract the last value for Ehomo from the last block of Alpha occ.
    if ehomo_blocks:
        ehomo_values = re.findall(r'-?\d+\.\d+', ehomo_blocks[-1])
        ehomo = float(ehomo_values[-1]) if ehomo_values else np.nan
    else:
        ehomo = np.nan
    
    # Extract the first value for Elumo from the last block of Alpha virt.
    if elumo_blocks:
        elumo_values = re.findall(r'-?\d+\.\d+', elumo_blocks[0])
        elumo = float(elumo_values[0]) if elumo_values else np.nan
    else:
        elumo = np.nan

    # Extract the %chk value
    chk_match = re.search(r'%chk=([^\s]+)', content)
    chk_value = chk_match.group(1) if chk_match else np.nan

    # Convert au to eV
    ehomo_ev = ehomo * 27.211396132 if not np.isnan(ehomo) else np.nan
    elumo_ev = elumo * 27.211396132 if not np.isnan(elumo) else np.nan
    
    # Calculate derived properties
    gap = elumo_ev - ehomo_ev if not np.isnan(elumo_ev) and not np.isnan(ehomo_ev) else np.nan
    ionisation_energy = -ehomo_ev if not np.isnan(ehomo_ev) else np.nan
    electron_affinity = -elumo_ev if not np.isnan(elumo_ev) else np.nan
    global_hardness = (elumo_ev - ehomo_ev) / 2.0 if not np.isnan(elumo_ev) and not np.isnan(ehomo_ev) else np.nan
    chem_potential = (elumo_ev + ehomo_ev) / 2.0 if not np.isnan(elumo_ev) and not np.isnan(ehomo_ev) else np.nan
    electrophilicity_index = (chem_potential ** 2) / (2 * global_hardness) if not np.isnan(chem_potential) and not np.isnan(global_hardness) else np.nan

    return {
        "Filename": os.path.basename(file_path),
        'name': chk_value,
        "homo_ev": ehomo_ev,
        "lumo_ev": elumo_ev,
        "gap_ev": gap,
        "ionization_ev": ionisation_energy,
        "elec_affinity_ev": electron_affinity,
        "hardness": global_hardness,
        "chem_potential": chem_potential,
        "electrophil_index": electrophilicity_index
    }

