from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler


class FeatureBuilder:
    def __init__(
        self,
        input_columns: list[str],
        features_col: str,
        scaled_features_col: str,
    ):
        self.input_columns = input_columns
        self.features_col = features_col
        self.scaled_features_col = scaled_features_col

    def assemble_features(self, df: DataFrame) -> DataFrame:
        assembler = VectorAssembler(
            inputCols=self.input_columns,
            outputCol=self.features_col,
        )

        return assembler.transform(df)

    def scale_features(self, df: DataFrame) -> DataFrame:
        scaler = StandardScaler(
            inputCol=self.features_col,
            outputCol=self.scaled_features_col,
            withStd=True,
            withMean=True,
        )

        scaler_model = scaler.fit(df)
        return scaler_model.transform(df)

    def build(self, df: DataFrame) -> DataFrame:
        featured_df = self.assemble_features(df)
        featured_df = self.scale_features(featured_df)

        return featured_df