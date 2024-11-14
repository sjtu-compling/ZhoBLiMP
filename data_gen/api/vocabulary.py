import pandas as pd
from io import StringIO


class Vocabulary(pd.DataFrame):

    @classmethod
    def from_string(cls, vocab_text):
        lines = vocab_text.replace("\t", ",")
        try:
            return cls(pd.read_csv(StringIO(lines), dtype=str))
        except pd.errors.EmptyDataError:
            raise ValueError("The vocabulary is empty.")
    
    @classmethod
    def from_csv(cls, vocab_fn):
        return cls(pd.read_csv(vocab_fn, dtype=str))

    def __init__(self, df):
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
        df = df[df.expression.notna()]

        if "expression" not in df.columns:
            raise ValueError("The vocabulary must have a column named 'expression'.")
        
        for col in df.columns.drop("expression"):
            # split the string by "/" and explode the dataframe 
            # to have one row for each value
            df[col] = df[col].str.split("/")
            df = df.explode(col)
        super().__init__(data=df)
        self._cache = {}

    def filter(self, property_kwargs, return_list=False):
        """Get all the lexical entries that match the given kwargs"""
        if not repr(property_kwargs) in self._cache:
            vocab = self.copy()
            for property, value in property_kwargs.items():
                if not isinstance(value, list):
                    value = [value]
                vocab = vocab[vocab[property].isin(value)]
            self._cache[repr(property_kwargs)] = Vocabulary(vocab)
        vocab = self._cache[repr(property_kwargs)]
        return vocab.expression.tolist() if return_list else vocab
    
    def get_values(self, property):
        """Get all the values of a given property"""
        return self[property].dropna().unique().tolist()

