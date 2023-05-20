import numpy as np
import pandas as pd
import psutil
import warnings
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import BayesianGaussianMixture
from collections import namedtuple

class BaseTransformer:
    """Base class for all transformers.
    The ``BaseTransformer`` class contains methods that must be implemented
    in order to create a new transformer. The ``fit`` method is optional,
    and ``fit_transform`` method is already implemented.
    """

    def fit(self, data):
        """Fit the transformer to the data.
        Args:
            data (pandas.Series or numpy.array):
                Data to transform.
        """
        raise NotImplementedError()

    def transform(self, data):
        """Transform the data.
        Args:
            data (pandas.Series or numpy.array):
                Data to transform.
        Returns:
            numpy.array:
                Transformed data.
        """
        raise NotImplementedError()

    def fit_transform(self, data):
        """Fit the transformer to the data and then transform it.
        Args:
            data (pandas.Series or numpy.array):
                Data to transform.
        Returns:
            numpy.array:
                Transformed data.
        """
        self.fit(data)
        return self.transform(data)

    def reverse_transform(self, data):
        """Revert the transformations to the original values.
        Args:
            data (pandas.Series or numpy.array):
                Data to transform.
        Returns:
            pandas.Series:
                Reverted data.
        """
        raise NotImplementedError()


IRREVERSIBLE_WARNING = (
    'Replacing nulls with existing value without `null_column`, which is not reversible. '
    'Use `null_column=True` to ensure that the transformation is reversible.'
)


class NullTransformer(BaseTransformer):
    """Transformer for data that contains Null values.
    Args:
        fill_value (object or None):
            Value to replace nulls. If ``None``, nans are not replaced.
        null_column (bool):
            Whether to create a new column to indicate which values were null or not.
            If ``None``, only create a new column when the data contains null values.
            If ``True``, always create the new column whether there are null values or not.
            If ``False``, do not create the new column.
            Defaults to ``None``.
        copy (bool):
            Whether to create a copy of the input data or modify it destructively.
    """

    nulls = None
    _null_column = None
    _fill_value = None

    def __init__(self, fill_value, null_column=None, copy=False):
        self.fill_value = fill_value
        self.null_column = null_column
        self.copy = copy

    def fit(self, data):
        """Fit the transformer to the data.
        Evaluate if the transformer has to create the null column or not.
        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.
        """
        null_values = data.isnull().values # Boolean df where True isnan
        self.nulls = null_values.any() # True if any True exists, False if no nulls in df
        contains_not_null = not null_values.all()
        if self.fill_value == 'mean':
            self._fill_value = data.mean() if contains_not_null else 0
        elif self.fill_value == 'mode':
            self._fill_value = data.mode(dropna=True)[0] if contains_not_null else 0
        else:
            self._fill_value = self.fill_value

        if self.null_column is None:
            self._null_column = self.nulls
        else:
            self._null_column = self.null_column

        if self._null_column:
            self.num_dt_cols = 2
        else:
            self.num_dt_cols = 1

    def transform(self, data):
        """Replace null values with the indicated fill_value.
        If required, create the null indicator column.
        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.
        Returns:
            numpy.ndarray
        """
        if self.nulls:
            if not isinstance(data, pd.Series):
                data = pd.Series(data)

            isnull = data.isnull()
            if self.nulls and self._fill_value is not None:
                if not self.copy:
                    data[isnull] = self._fill_value
                else:
                    data = data.fillna(self._fill_value)

            if self._null_column:
                return pd.concat([data, isnull.astype('int')], axis=1).values

            if self._fill_value in data.values:
                warnings.warn(IRREVERSIBLE_WARNING)

        reshaped_data = data.values.reshape(-1, 1)
        return reshaped_data

    def reverse_transform(self, data):
        """Restore null values to the data.
        If a null indicator column was created during fit, use it as a reference.
        Otherwise, replace all instances of ``fill_value`` that can be found in
        data.
        Args:
            data (numpy.ndarray):
                Data to transform.
        Returns:
            pandas.Series
        """
        if self.nulls:
            if self._null_column:
                isnull = data[:, 1] > 0.5
                data = pd.Series(data[:, 0])
            else:
                isnull = np.where(self._fill_value == data)[0]
                data = pd.Series(data)

            if isnull.any():
                if self.copy:
                    data = data.copy()

                data.iloc[isnull] = np.nan

        return data

# class CategoricalTransformer(BaseTransformer):
#     """Transformer for categorical data.
#     This transformer computes a float representative for each one of the categories
#     found in the fit data, and then replaces the instances of these categories with
#     the corresponding representative.
#     The representatives are decided by sorting the categorical values by their relative
#     frequency, then dividing the ``[0, 1]`` interval by these relative frequencies, and
#     finally assigning the middle point of each interval to the corresponding category.
#     When the transformation is reverted, each value is assigned the category that
#     corresponds to the interval it falls in.
#     Null values are considered just another category.
#     Args:
#         fuzzy (bool):
#             Whether to generate gaussian noise around the class representative of each interval
#             or just use the mean for all the replaced values. Defaults to ``False``.
#         clip (bool):
#             If ``True``, clip the values to [0, 1]. Otherwise normalize them using modulo 1.
#             Defaults to ``False``.
#     """

#     mapping = None
#     intervals = None
#     starts = None
#     means = None
#     dtype = None
#     _get_category_from_index = None

#     def __setstate__(self, state):
#         """Replace any ``null`` key by the actual ``np.nan`` instance."""
#         intervals = state.get('intervals')
#         if intervals:
#             for key in list(intervals):
#                 if pd.isnull(key):
#                     intervals[np.nan] = intervals.pop(key)

#         self.__dict__ = state

#     def __init__(self, fuzzy=False, clip=False):
#         self.fuzzy = fuzzy
#         self.clip = clip

#     @staticmethod
#     def _get_intervals(data):
#         """Compute intervals for each categorical value.
#         Args:
#             data (pandas.Series):
#                 Data to analyze.
#         Returns:
#             dict:
#                 intervals for each categorical value (start, end).
#         """
#         frequencies = data.value_counts(dropna=False)

#         start = 0
#         end = 0
#         elements = len(data)

#         intervals = {}
#         means = []
#         starts = []
#         for value, frequency in frequencies.items():
#             prob = frequency / elements
#             end = start + prob
#             mean = start + prob / 2
#             std = prob / 6
#             if pd.isnull(value):
#                 value = np.nan

#             intervals[value] = (start, end, mean, std)
#             means.append(mean)
#             starts.append((value, start))
#             start = end

#         means = pd.Series(means, index=list(frequencies.keys()))
#         starts = pd.DataFrame(starts, columns=['category', 'start']).set_index('start')

#         return intervals, means, starts

#     def fit(self, data):
#         """Fit the transformer to the data.
#         Create the mapping dict to save the label encoding.
#         Finally, compute the intervals for each categorical value.
#         Args:
#             data (pandas.Series or numpy.ndarray):
#                 Data to fit the transformer to.
#         """
#         self.mapping = {}
#         self.dtype = data.dtype

#         if isinstance(data, np.ndarray):
#             data = pd.Series(data)

#         self.intervals, self.means, self.starts = self._get_intervals(data)
#         self._get_category_from_index = list(self.means.index).__getitem__

#     def _transform_by_category(self, data):
#         """Transform the data by iterating over the different categories."""
#         result = np.empty(shape=(len(data), ), dtype=float)

#         # loop over categories
#         for category, values in self.intervals.items():
#             mean, std = values[2:]
#             if category is np.nan:
#                 mask = data.isnull()
#             else:
#                 mask = (data.values == category)

#             if self.fuzzy:
#                 result[mask] = norm.rvs(mean, std, size=mask.sum())
#             else:
#                 result[mask] = mean

#         return result

#     def _get_value(self, category):
#         """Get the value that represents this category."""
#         if pd.isnull(category):
#             category = np.nan

#         mean, std = self.intervals[category][2:]

#         if self.fuzzy:
#             return norm.rvs(mean, std)

#         return mean

#     def _transform_by_row(self, data):
#         """Transform the data row by row."""
#         return data.fillna(np.nan).apply(self._get_value).to_numpy()

#     def transform(self, data):
#         """Transform categorical values to float values.
#         Replace the categories with their float representative value.
#         Args:
#             data (pandas.Series or numpy.ndarray):
#                 Data to transform.
#         Returns:
#             numpy.ndarray:
#         """
#         if not isinstance(data, pd.Series):
#             data = pd.Series(data)

#         if len(self.means) < len(data):
#             return self._transform_by_category(data)

#         return self._transform_by_row(data)

#     def _normalize(self, data):
#         """Normalize data to the range [0, 1].
#         This is done by either clipping or computing the values modulo 1.
#         """
#         if self.clip:
#             return data.clip(0, 1)

#         return np.mod(data, 1)

#     def _reverse_transform_by_matrix(self, data):
#         """Reverse transform the data with matrix operations."""
#         num_rows = len(data)
#         num_categories = len(self.means)

#         data = np.broadcast_to(data, (num_categories, num_rows)).T
#         means = np.broadcast_to(self.means, (num_rows, num_categories))
#         diffs = np.abs(np.subtract(data, means))
#         indexes = np.argmin(diffs, axis=1)

#         self._get_category_from_index = list(self.means.index).__getitem__
#         return pd.Series(indexes).apply(self._get_category_from_index).astype(self.dtype)

#     def _reverse_transform_by_category(self, data):
#         """Reverse transform the data by iterating over all the categories."""
#         result = np.empty(shape=(len(data), ), dtype=self.dtype)

#         # loop over categories
#         for category, values in self.intervals.items():
#             start = values[0]
#             mask = (start <= data.values)
#             result[mask] = category

#         return pd.Series(result, index=data.index, dtype=self.dtype)

#     def _get_category_from_start(self, value):
#         lower = self.starts.loc[:value]
#         return lower.iloc[-1].category

#     def _reverse_transform_by_row(self, data):
#         """Reverse transform the data by iterating over each row."""
#         return data.apply(self._get_category_from_start).astype(self.dtype)

#     def reverse_transform(self, data):
#         """Convert float values back to the original categorical values.
#         Args:
#             data (numpy.ndarray):
#                 Data to revert.
#         Returns:
#             pandas.Series
#         """
#         if not isinstance(data, pd.Series):
#             if len(data.shape) > 1:
#                 data = data[:, 0]

#             data = pd.Series(data)

#         data = self._normalize(data)

#         num_rows = len(data)
#         num_categories = len(self.means)

#         # total shape * float size * number of matrices needed
#         needed_memory = num_rows * num_categories * 8 * 3
#         available_memory = psutil.virtual_memory().available
#         if available_memory > needed_memory:
#             return self._reverse_transform_by_matrix(data)

#         if num_rows > num_categories:
#             return self._reverse_transform_by_category(data)

#         # loop over rows
#         return self._reverse_transform_by_row(data)


class DatetimeTransformer(BaseTransformer):
    """Transformer for datetime data.
    This transformer replaces datetime values with an integer timestamp
    transformed to float.
    Null values are replaced using a ``NullTransformer``.
    Args:
        nan (int, str or None):
            Indicate what to do with the null values. If an integer is given, replace them
            with the given value. If the strings ``'mean'`` or ``'mode'`` are given, replace
            them with the corresponding aggregation. If ``None`` is given, do not replace them.
            Defaults to ``'mean'``.
        null_column (bool):
            Whether to create a new column to indicate which values were null or not.
            If ``None``, only create a new column when the data contains null values.
            If ``True``, always create the new column whether there are null values or not.
            If ``False``, do not create the new column.
            Defaults to ``None``.
        strip_constant (bool):
            Whether to optimize the output values by finding the smallest time unit that
            is not zero on the training datetimes and dividing the generated numerical
            values by the value of the next smallest time unit. This, a part from reducing the
            orders of magnitued of the transformed values, ensures that reverted values always
            are zero on the lower time units.
    """

    null_transformer = None
    divider = None

    # null_column hyperparameter tuning below
    # Keep on False for now. Debug later (low priority)
    def __init__(self, nan='mean', null_column=False, strip_constant=False):
        self.nan = nan
        self.null_column = null_column
        self.strip_constant = strip_constant

    def _find_divider(self, transformed):
        self.divider = 1
        multipliers = [10] * 9 + [60, 60, 24]
        for multiplier in multipliers:
            candidate = self.divider * multiplier
            if np.mod(transformed, candidate).any():
                break

            self.divider = candidate

    def _transform(self, datetimes):
        """Transform datetime values to integer."""
        nulls = datetimes.isnull()
        integers = pd.to_numeric(datetimes, errors='coerce').values.astype(np.float64)
        integers[nulls] = np.nan
        transformed = pd.Series(integers)

        if self.strip_constant:
            self._find_divider(transformed)
            transformed = transformed.floordiv(self.divider)


        return transformed

    def fit(self, data):
        """Fit the transformer to the data.
        Args:
            data (pandas.Series or numpy.ndarray):
                Data to fit the transformer to.
        """
        if isinstance(data, np.ndarray):
            #data = data.reshape(-1)
            data = pd.Series(data)

        transformed = self._transform(data)
        self.null_transformer = NullTransformer(self.nan, self.null_column, copy=True)
        self.null_transformer.fit(transformed)
        self.num_dt_cols = self.null_transformer.num_dt_cols

    def transform(self, data):
        """Transform datetime values to float values.
        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.
        Returns:
            numpy.ndarray
        """
        if isinstance(data, np.ndarray):
            data = data.reshape(-1)
            data = pd.Series(data)

        data = self._transform(data)

        return self.null_transformer.transform(data)

    def reverse_transform(self, data):
        """Convert float values back to datetimes.
        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.
        Returns:
            pandas.Series
        """
        if self.nan is not None:
            data = self.null_transformer.reverse_transform(data)

        if isinstance(data, np.ndarray) and (data.ndim == 2):
            data = data[:, 0]

        data = np.round(data.astype(np.float64))
        if self.strip_constant:
            data = data * self.divider

        return pd.to_datetime(data)


# class OneHotEncodingTransformer(BaseTransformer):
#     """OneHotEncoding for categorical data.
#     This transformer replaces a single vector with N unique categories in it
#     with N vectors which have 1s on the rows where the corresponding category
#     is found and 0s on the rest.
#     Null values are considered just another category.
#     Args:
#         error_on_unknown (bool):
#             If a value that was not seen during the fit stage is passed to
#             transform, then an error will be raised if this is True.
#     """

#     dummies = None
#     _dummy_na = None
#     _num_dummies = None
#     _dummy_encoded = False
#     _indexer = None
#     _uniques = None

#     def __init__(self, error_on_unknown=True):
#         self.error_on_unknown = error_on_unknown

#     @staticmethod
#     def _prepare_data(data):
#         """Transform data to appropriate format.
#         If data is a valid list or a list of lists, transforms it into an np.array,
#         otherwise returns it.
#         Args:
#             data (pandas.Series, numpy.ndarray, list or list of lists):
#                 Data to prepare.
#         Returns:
#             pandas.Series or numpy.ndarray
#         """
#         if isinstance(data, list):
#             data = np.array(data)

#         if len(data.shape) > 2:
#             raise ValueError('Unexpected format.')
#         if len(data.shape) == 2:
#             if data.shape[1] != 1:
#                 raise ValueError('Unexpected format.')

#             data = data[:, 0]

#         return data

#     def _transform(self, data):
#         if self._dummy_encoded:
#             coder = self._indexer
#             codes = pd.Categorical(data, categories=self._uniques).codes
#         else:
#             coder = self._uniques
#             codes = data

#         rows = len(data)
#         dummies = np.broadcast_to(coder, (rows, self._num_dummies))
#         coded = np.broadcast_to(codes, (self._num_dummies, rows)).T
#         array = (coded == dummies).astype(int)

#         if self._dummy_na:
#             null = np.zeros((rows, 1), dtype=int)
#             null[pd.isnull(data)] = 1
#             array = np.append(array, null, axis=1)

#         return array

#     def fit(self, data):
#         """Fit the transformer to the data.
#         Get the pandas `dummies` which will be used later on for OneHotEncoding.
#         Args:
#             data (pandas.Series, numpy.ndarray, list or list of lists):
#                 Data to fit the transformer to.
#         """
#         data = self._prepare_data(data)

#         null = pd.isnull(data)
#         self._uniques = list(pd.unique(data[~null]))
#         self._dummy_na = null.any()
#         self._num_dummies = len(self._uniques)
#         self._indexer = list(range(self._num_dummies))
#         self.dummies = self._uniques.copy()

#         if not np.issubdtype(data.dtype, np.number):
#             self._dummy_encoded = True

#         if self._dummy_na:
#             self.dummies.append(np.nan)

#     def transform(self, data):
#         """Replace each category with the OneHot vectors.
#         Args:
#             data (pandas.Series, numpy.ndarray, list or list of lists):
#                 Data to transform.
#         Returns:
#             numpy.ndarray:
#         """
#         data = self._prepare_data(data)
#         array = self._transform(data)

#         if self.error_on_unknown:
#             unknown = array.sum(axis=1) == 0
#             if unknown.any():
#                 raise ValueError(f'Attempted to transform {list(data[unknown])} ',
#                                  'that were not seen during fit stage.')

#         return array

#     def reverse_transform(self, data):
#         """Convert float values back to the original categorical values.
#         Args:
#             data (numpy.ndarray):
#                 Data to revert.
#         Returns:
#             pandas.Series
#         """
#         if data.ndim == 1:
#             data = data.reshape(-1, 1)

#         indices = np.argmax(data, axis=1)
#         return pd.Series(indices).map(self.dummies.__getitem__)


class LabelEncodingTransformer(BaseTransformer):
    """LabelEncoding for categorical data.
    This transformer generates a unique integer representation for each category
    and simply replaces each category with its integer value.
    Null values are considered just another category.
    Attributes:
        values_to_categories (dict):
            Dictionary that maps each integer value for its category.
        categories_to_values (dict):
            Dictionary that maps each category with the corresponding
            integer value.
    """

    values_to_categories = None
    categories_to_values = None

    def fit(self, data):
        """Fit the transformer to the data.
        Generate a unique integer representation for each category and
        store them in the `categories_to_values` dict and its reverse
        `values_to_categories`.
        Args:
            data (pandas.Series or numpy.ndarray):
                Data to fit the transformer to.
        """
        self.values_to_categories = dict(enumerate(pd.unique(data)))
        self.categories_to_values = {
            category: value
            for value, category in self.values_to_categories.items()
        }

    def transform(self, data):
        """Replace each category with its corresponding integer value.
        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.
        Returns:
            numpy.ndarray:
        """
        if not isinstance(data, pd.Series):
            data = data.reshape(-1)
            data = pd.Series(data)

        transformed = data.map(self.categories_to_values)

        transformed_reshaped = transformed.values.reshape(-1,1)

        return transformed_reshaped

    def reverse_transform(self, data):
        """Convert float values back to the original categorical values.
        Args:
            data (numpy.ndarray):
                Data to revert.
        Returns:
            pandas.Series
        """
        if isinstance(data, np.ndarray) and (data.ndim == 2):
            data = data[:, 0]

        data = data.clip(min(self.values_to_categories), max(self.values_to_categories))
        return pd.Series(data).round().map(self.values_to_categories)

##############################################################################################################


SpanInfo = namedtuple("SpanInfo", ["dim", "activation_fn"])
ColumnTransformInfo = namedtuple(
    "ColumnTransformInfo", ["column_name", "column_type",
                            "transform", "transform_aux", "unique_cats",
                            "output_info", "output_dimensions"])

#GAN_ColumnInfo = namedtuple("GAN_ColumnInfo", ["column_name", "column_gan_type", "start_idx", "stop_idx", "dim"])
GAN_ColumnInfo = namedtuple("GAN_ColumnInfo", ["column_name", "column_gan_type", "column_index", "column_unique_cats", "column_dim", "column_span_out_info"])

class DataPrep:
    """Data Transformer.
    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    def __init__(self, vgm_mode = False):
        """Create a data transformer.
        """
        self.vgm_mode = vgm_mode
        self._dummy_placeholder = 1
        super(DataPrep, self).__init__()
        self._max_clusters = 10
        self._weight_threshold = 0.05
        
    def _fit_continuous(self, column_name, raw_column_data):
        """Fit minmax scaler for numeric columns."""

        raw_column_data = raw_column_data.reshape(-1, 1)

        minmax = MinMaxScaler()
        minmax.fit(raw_column_data)

        return ColumnTransformInfo(
            column_name=column_name, column_type="continuous", transform=minmax,
            transform_aux=None,
            unique_cats = 1,
            output_info=[SpanInfo(1, 'tanh')],
            output_dimensions=1)

    def _fit_continuous_vgm(self, column_name, raw_column_data):
        """Fit VGM for numeric columns."""

        gm = BayesianGaussianMixture(
            n_components=self._max_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            n_init=1
        )

        gm.fit(raw_column_data.reshape(-1, 1))
        valid_component_indicator = gm.weights_ > self._weight_threshold
        num_components = valid_component_indicator.sum()


        return ColumnTransformInfo(
            column_name=column_name, column_type="continuous", transform=gm,
            transform_aux=valid_component_indicator,
            unique_cats = 1+num_components,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions= 1+num_components)


    # def _fit_discrete(self, column_name, raw_column_data):
    #     """Fit one hot encoder for discrete column."""
    #     ohe = OneHotEncodingTransformer()
    #     ohe.fit(raw_column_data)
    #     num_categories = len(ohe.dummies)

    #     return ColumnTransformInfo(
    #         column_name=column_name, column_type="discrete", transform=ohe,
    #         transform_aux=None,
    #         output_info=[SpanInfo(num_categories, 'softmax')],
    #         output_dimensions=num_categories)

    def _fit_discrete(self, column_name, raw_column_data):
        """Fit one hot encoder for discrete column."""
        lbe = LabelEncodingTransformer()
        lbe.fit(raw_column_data)
        num_categories = len(lbe.values_to_categories)

        return ColumnTransformInfo(
            column_name=column_name, column_type="discrete", transform=lbe,
            transform_aux=None,
            unique_cats = num_categories,
            output_info=[SpanInfo(1, 'softmax')],
            output_dimensions=1)

    def _fit_datetime(self, column_name, raw_column_data):
        """Fit datetime transformer."""

        datetf = DatetimeTransformer()
        datetf.fit(raw_column_data)
        num_dt_cols = datetf.num_dt_cols

        return ColumnTransformInfo(
            column_name=column_name, column_type="datetime", transform=datetf,
            transform_aux=None,
            unique_cats = 1,
            output_info=[SpanInfo(num_dt_cols, 'tanh')],
            output_dimensions=num_dt_cols)

    def fit(self, raw_data, discrete_columns=tuple(), datetime_columns = tuple()):
        """Fit GMM for continuous columns and One hot encoder for discrete columns.
        This step also counts the #columns in matrix data, and span information.
        """
        self.output_info_list = []
        self.gan_ColumnInfo = []
        self.output_dimensions = 0
        iter_idx = 0

        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            raw_data = pd.DataFrame(raw_data)
        else:
            self.dataframe = True

        self._column_raw_dtypes = raw_data.infer_objects().dtypes

        self._column_transform_info_list = []
        for column_name in raw_data.columns:
            raw_column_data = raw_data[column_name].values
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(
                    column_name, raw_column_data)
            elif column_name in datetime_columns:
                column_transform_info = self._fit_datetime(
                    column_name, raw_column_data)
            else:
                if self.vgm_mode:
                    column_transform_info = self._fit_continuous_vgm(
                        column_name, raw_column_data)
                else:
                    column_transform_info = self._fit_continuous(
                        column_name, raw_column_data)

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

        # for item in self._column_transform_info_list:
        #     gan_col_name = item.column_name
        #     gan_col_type = item.column_type
        #     if gan_col_type == "discrete":
        #         gan_col_type = "categorical"
        #     else:
        #         gan_col_type = "numerical"
        #     gan_start_idx = iter_idx
        #     gan_temp_dim = item.output_info[0].dim
        #     gan_stop_idx = gan_start_idx + gan_temp_dim
        #     iter_idx = gan_stop_idx

        #     item_gan_col_info = GAN_ColumnInfo(
        #         column_name = gan_col_name,
        #         column_gan_type = gan_col_type,
        #         start_idx = gan_start_idx,
        #         stop_idx = gan_stop_idx,
        #         dim = gan_temp_dim )

        #     self.gan_ColumnInfo.append(item_gan_col_info)

        for item in self._column_transform_info_list:
            gan_col_name = item.column_name
            gan_col_type = item.column_type
            if gan_col_type == "discrete":
                gan_col_type = "categorical"
            else:
                gan_col_type = "numerical"
            gan_col_idx = iter_idx
            gan_unq_dim = item.unique_cats
            col_dim = item.output_dimensions
            output_span_info = item.output_info

            item_gan_col_info = GAN_ColumnInfo(
                column_name = gan_col_name,
                column_gan_type = gan_col_type,
                column_index = gan_col_idx,
                column_unique_cats = gan_unq_dim,
                column_dim = col_dim,
                column_span_out_info = output_span_info)

            self.gan_ColumnInfo.append(item_gan_col_info)

            iter_idx += item.output_info[0].dim


    def _transform_continuous(self, column_transform_info, raw_column_data):
        minmax = column_transform_info.transform
        return [minmax.transform(raw_column_data)]

    def _transform_continuous_vgm(self, column_transform_info, raw_column_data):
        gm = column_transform_info.transform

        valid_component_indicator = column_transform_info.transform_aux
        num_components = valid_component_indicator.sum()

        means = gm.means_.reshape((1, self._max_clusters))
        stds = np.sqrt(gm.covariances_).reshape((1, self._max_clusters))
        normalized_values = ((raw_column_data - means) / (4 * stds))[:, valid_component_indicator]
        component_probs = gm.predict_proba(raw_column_data)[:, valid_component_indicator]

        selected_component = np.zeros(len(raw_column_data), dtype='int')
        for i in range(len(raw_column_data)):
            component_porb_t = component_probs[i] + 1e-6
            component_porb_t = component_porb_t / component_porb_t.sum()
            selected_component[i] = np.random.choice(
                np.arange(num_components), p=component_porb_t)

        selected_normalized_value = normalized_values[
            np.arange(len(raw_column_data)), selected_component].reshape([-1, 1])
        selected_normalized_value = np.clip(selected_normalized_value, -.99, .99)

        selected_component_onehot = np.zeros_like(component_probs)
        selected_component_onehot[np.arange(len(raw_column_data)), selected_component] = 1
        return [selected_normalized_value, selected_component_onehot]

    # def _transform_discrete(self, column_transform_info, raw_column_data):
    #     ohe = column_transform_info.transform
    #     return [ohe.transform(raw_column_data)]

    def _transform_discrete(self, column_transform_info, raw_column_data):
        lbe = column_transform_info.transform
        return [lbe.transform(raw_column_data)]

    def _transform_datetime(self, column_transform_info, raw_column_data):
        datetf = column_transform_info.transform
        return [datetf.transform(raw_column_data)]

    def transform(self, raw_data):
        """Take raw data and output a matrix data."""
        if not isinstance(raw_data, pd.DataFrame):
            raw_data = pd.DataFrame(raw_data)

        column_data_list = []
        for column_transform_info in self._column_transform_info_list:
            column_data = raw_data[[column_transform_info.column_name]].values
            if column_transform_info.column_type == "continuous":
                if self.vgm_mode:
                    column_data_list += self._transform_continuous_vgm(
                        column_transform_info, column_data)
                else:
                    column_data_list += self._transform_continuous(
                        column_transform_info, column_data)
            elif column_transform_info.column_type == "datetime":
                column_data_list += self._transform_datetime(
                    column_transform_info, column_data)
            else:
                assert column_transform_info.column_type == "discrete"
                column_data_list += self._transform_discrete(
                    column_transform_info, column_data)
        
        return np.concatenate(column_data_list, axis=1).astype(float)

    def _inverse_transform_continuous(self, column_transform_info, column_data):
        minmax = column_transform_info.transform
        return minmax.inverse_transform(column_data)


    def _inverse_transform_continuous_vgm(self, column_transform_info, column_data):
        gm = column_transform_info.transform
        valid_component_indicator = column_transform_info.transform_aux

        selected_normalized_value = column_data[:, 0]
        selected_component_probs = column_data[:, 1:]

        selected_normalized_value = np.clip(selected_normalized_value, -1, 1)
        component_probs = np.ones((len(column_data), self._max_clusters)) * -100
        component_probs[:, valid_component_indicator] = selected_component_probs

        means = gm.means_.reshape([-1])
        stds = np.sqrt(gm.covariances_).reshape([-1])
        selected_component = np.argmax(component_probs, axis=1)

        std_t = stds[selected_component]
        mean_t = means[selected_component]
        column = selected_normalized_value * 4 * std_t + mean_t

        return column
        
    # def _inverse_transform_discrete(self, column_transform_info, column_data):
    #     ohe = column_transform_info.transform
    #     return ohe.reverse_transform(column_data)

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        lbe = column_transform_info.transform
        return lbe.reverse_transform(column_data)
    
    def _inverse_transform_datetime(self, column_transform_info, column_data):
        datetf = column_transform_info.transform
        return datetf.reverse_transform(column_data)

    def inverse_transform(self, data, sigmas=None):
        """Take matrix data and output raw data.
        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        recovered_data = pd.DataFrame()


        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st:st + dim]

            if column_transform_info.column_type == 'continuous':
                if self.vgm_mode:
                    recovered_column_data = self._inverse_transform_continuous_vgm(
                        column_transform_info, column_data)
                    recovered_data[column_transform_info.column_name] = recovered_column_data.reshape(-1)
                    recovered_data[column_transform_info.column_name] = recovered_data[column_transform_info.column_name].astype(self._column_raw_dtypes[column_transform_info.column_name])
                else:
                    recovered_column_data = self._inverse_transform_continuous(
                        column_transform_info, column_data)
                    recovered_data[column_transform_info.column_name] = recovered_column_data.reshape(-1)
                    recovered_data[column_transform_info.column_name] = np.around(recovered_data[column_transform_info.column_name]).astype(self._column_raw_dtypes[column_transform_info.column_name])
            elif column_transform_info.column_type == 'datetime':
                recovered_column_data = self._inverse_transform_datetime(
                    column_transform_info, column_data)
                recovered_data[column_transform_info.column_name] = recovered_column_data.tolist()
                recovered_data[column_transform_info.column_name] = recovered_data[column_transform_info.column_name].astype(self._column_raw_dtypes[column_transform_info.column_name])
            else:
                assert column_transform_info.column_type == 'discrete'
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data)
                recovered_data[column_transform_info.column_name] = recovered_column_data.tolist()
                recovered_data[column_transform_info.column_name] = recovered_data[column_transform_info.column_name].astype(self._column_raw_dtypes[column_transform_info.column_name])

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        # recovered_data = np.column_stack(recovered_column_data_list)
        # recovered_data = (pd.DataFrame(recovered_data, columns=column_names)
        #                   .astype(self._column_raw_dtypes))
        if not self.dataframe:
            recovered_data = recovered_data.values

        return recovered_data

    def convert_column_name_value_to_id(self, column_name, value):
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == "discrete":
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        one_hot = column_transform_info.transform.transform(np.array([value]))[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            "discrete_column_id": discrete_counter,
            "column_id": column_id,
            "value_id": np.argmax(one_hot)
        }