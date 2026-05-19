from pyspark.sql import DataFrame
from pyspark.sql.functions import col


class DataPreprocessor:
    def __init__(
        self,
        selected_columns: list[str],
        row_limit: int,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ):
        self.selected_columns = selected_columns
        self.row_limit = row_limit
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def select_columns(self, df: DataFrame) -> DataFrame:
        missing_columns = [
            column_name
            for column_name in self.selected_columns
            if column_name not in df.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing columns in dataframe: {missing_columns}")

        return df.select(*self.selected_columns)

    def remove_missing_values(self, df: DataFrame) -> DataFrame:
        return df.dropna()

    def remove_negative_values(self, df: DataFrame) -> DataFrame:
        condition = None

        for column_name in self.selected_columns:
            current_condition = col(column_name) >= 0
            condition = current_condition if condition is None else condition & current_condition

        return df.filter(condition)

    def remove_basic_outliers(self, df: DataFrame) -> DataFrame:
        return df.filter(
            (col("energy_100g") < 5000)
            & (col("fat_100g") < 100)
            & (col("carbohydrates_100g") < 100)
            & (col("sugars_100g") < 100)
            & (col("proteins_100g") < 100)
        )

    def remove_outliers_percentile(self, df: DataFrame) -> DataFrame:
        condition = None

        for column_name in self.selected_columns:
            lower, upper = df.approxQuantile(
                column_name,
                [self.lower_quantile, self.upper_quantile],
                0.01,
            )

            current_condition = (
                (col(column_name) >= lower)
                & (col(column_name) <= upper)
            )

            condition = current_condition if condition is None else condition & current_condition

        return df.filter(condition)

    def limit_rows(self, df: DataFrame) -> DataFrame:
        return df.limit(self.row_limit)

    def preprocess(self, df: DataFrame) -> DataFrame:
        processed_df = self.select_columns(df)
        processed_df = self.remove_missing_values(processed_df)
        processed_df = self.remove_negative_values(processed_df)

        processed_df = self.remove_basic_outliers(processed_df)
        processed_df = self.remove_outliers_percentile(processed_df)

        processed_df = self.limit_rows(processed_df)

        return processed_df