import datetime
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError


class DatabaseManager:
    """
    Manages database connections and operations for the weather data pipeline
    """

    def __init__(self):
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
        self._connection_string = self._get_db_connection_string()
        self._engine = None

    def _get_db_connection_string(self) -> str | None:
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")

        if not all([db_name, db_user, db_password]):
            print(
                "DB_Manager Error: Database credentials (DB_NAME, DB_USER, DB_PASSWORD) "
                "are not fully set in environment variables"
            )
            return None
        return f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    def _get_engine(self):
        if self._engine is None:
            if not self._connection_string:
                raise ValueError(
                    "DB connection string is not set. Cannot create engine."
                )
            self._engine = create_engine(self._connection_string)
            print("DB_Manager: Database engine created.")
        return self._engine

    def _dispose_engine(self):
        if self._engine:
            self._engine.dispose()
            self._engine = None
            print("DB_Manager: Database engine disposed.")

    def store_dataframe(
        self,
        data_to_store: pd.DataFrame,
        table_name: str,
        if_exists: str = "append",
        add_timestamp_col: str | None = None,
    ):
        """
        Stores a pandas DataFrame into a PostgreSQL table.

        Args:
            data_to_store (pd.DataFrame): The DataFrame to store.
            table_name (str): The name of the table in the database.
            if_exists (str): How to behave if the table already exists.
            add_timestamp_col (str | None): If provided, adds a column with the current UTC timestamp
                                        before storing. E.g., 'loaded_at', 'generated_at'.
        """
        if data_to_store is None or data_to_store.empty:
            print(
                f"DB_Manager: DataFrame for table '{table_name}' is empty or None. Skipping storage."
            )
            return

        try:
            engine = self._get_engine()

            df_to_store = (
                data_to_store.reset_index()
                if "date" in data_to_store.index.names
                else data_to_store
            )

            if add_timestamp_col:
                df_to_store[add_timestamp_col] = datetime.datetime.now(
                    datetime.timezone.utc
                )

            df_to_store.to_sql(
                table_name, engine, if_exists=if_exists, index=False, chunksize=1000
            )
            print(
                f"DB_Manager: DataFrame successfully stored to table '{table_name}' "
                f"(if_exists='{if_exists}')."
            )
        except SQLAlchemyError as e:
            print(f"DB_Manager Error: Storing dataframe to database: {e}")
        except Exception as e:
            print(f"DB_Manager Unexpected Error: {e}")

    def load_dataframe(self, table_name: str) -> pd.DataFrame:
        """
        Loads data from a specified PostgreSQL table into a Pandas DataFrame.
        """
        try:
            engine = self._get_engine()
            with engine.connect() as connection:
                df = pd.read_sql_table(table_name, connection)
                print(
                    f"DB_Manager: Successfully loaded {len(df)} rows from table '{table_name}'."
                )

                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date").reset_index(drop=True)
                else:
                    print(
                        f"DB_Manager Warning: 'date' column not found in table '{table_name}'."
                    )

                return df
        except SQLAlchemyError as e:
            print(f"DB_Manager Error: Loading data from '{table_name}': {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"DB_Manager Unexpected Error: {e}")
            return pd.DataFrame()
