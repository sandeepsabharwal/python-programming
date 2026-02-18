"""
MyNumpy - A Custom Matrix Library
=================================
A comprehensive matrix library that provides functionality similar to numpy.
Supports all basic and advanced mathematical operations on matrices.

Author: Your Name
Version: 1.0.0
"""

import random
import math


class Matrix:
    """
    A matrix class that supports various mathematical operations.
    Uses lists internally to store array data.
    """
    
    def __init__(self, data=None, rows=None, cols=None):
        """
        Initialize a Matrix.
        
        Parameters:
        -----------
        data : list or nested list, optional
            Input data to create matrix from. Can be:
            - A flat list (requires rows and cols)
            - A nested list (2D array)
        rows : int, optional
            Number of rows (if data is flat list)
        cols : int, optional
            Number of columns (if data is flat list)
        
        Examples:
        ---------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m = Matrix([1, 2, 3, 4], rows=2, cols=2)
        >>> m = Matrix()  # Creates empty matrix
        """
        self.data = []
        self.rows = 0
        self.cols = 0
        
        if data is not None:
            if isinstance(data[0], list):
                # 2D list input
                self.data = [row[:] for row in data]
                self.rows = len(data)
                self.cols = len(data[0]) if data else 0
            else:
                # Flat list input
                if rows and cols:
                    self.rows = rows
                    self.cols = cols
                    self.data = [data[i * cols:(i + 1) * cols] for i in range(rows)]
                else:
                    raise ValueError("Rows and columns must be specified for flat list input")
    
    # ==================== Static Factory Methods ====================
    
    @staticmethod
    def zeros(rows, cols):
        """
        Create a matrix filled with zeros.
        
        Parameters:
        -----------
        rows : int
            Number of rows
        cols : int
            Number of columns
            
        Returns:
        --------
        Matrix
            A matrix filled with zeros
            
        Examples:
        ---------
        >>> Matrix.zeros(2, 3)
        """
        return Matrix([[0 for _ in range(cols)] for _ in range(rows)])
    
    @staticmethod
    def ones(rows, cols):
        """
        Create a matrix filled with ones.
        
        Parameters:
        -----------
        rows : int
            Number of rows
        cols : int
            Number of columns
            
        Returns:
        --------
        Matrix
            A matrix filled with ones
        """
        return Matrix([[1 for _ in range(cols)] for _ in range(rows)])
    
    @staticmethod
    def identity(size):
        """
        Create an identity matrix.
        
        Parameters:
        -----------
        size : int
            Size of the square matrix (size x size)
            
        Returns:
        --------
        Matrix
            An identity matrix of given size
        """
        data = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
        return Matrix(data)
    
    @staticmethod
    def random(rows, cols, min_val=0, max_val=1):
        """
        Create a matrix with random values.
        
        Parameters:
        -----------
        rows : int
            Number of rows
        cols : int
            Number of columns
        min_val : float, optional
            Minimum random value (default: 0)
        max_val : float, optional
            Maximum random value (default: 1)
            
        Returns:
        --------
        Matrix
            A matrix with random values
        """
        data = [[random.uniform(min_val, max_val) for _ in range(cols)] for _ in range(rows)]
        return Matrix(data)
    
    @staticmethod
    def random_int(rows, cols, min_val=0, max_val=10):
        """
        Create a matrix with random integer values.
        
        Parameters:
        -----------
        rows : int
            Number of rows
        cols : int
            Number of columns
        min_val : int, optional
            Minimum random value (default: 0)
        max_val : int, optional
            Maximum random value (default: 10)
            
        Returns:
        --------
        Matrix
            A matrix with random integer values
        """
        data = [[random.randint(min_val, max_val) for _ in range(cols)] for _ in range(rows)]
        return Matrix(data)
    
    @staticmethod
    def diagonal(values):
        """
        Create a diagonal matrix from a list of values.
        
        Parameters:
        -----------
        values : list
            Values for the diagonal
            
        Returns:
        --------
        Matrix
            A diagonal matrix
        """
        size = len(values)
        data = [[values[i] if i == j else 0 for j in range(size)] for i in range(size)]
        return Matrix(data)
    
    # ==================== Input Methods ====================
    
    @classmethod
    def from_input(cls, prompt="Enter matrix data row by row (space-separated): "):
        """
        Create a matrix from user input.
        
        Parameters:
        -----------
        prompt : str, optional
            Prompt message for user
            
        Returns:
        --------
        Matrix
            A matrix created from user input
            
        Examples:
        ---------
        >>> m = Matrix.from_input("Enter values: ")
        """
        print(prompt)
        print("Enter each row on a new line, values separated by spaces.")
        print("Press Enter twice to finish.")
        
        rows = []
        while True:
            try:
                line = input().strip()
                if not line:
                    break
                row = [float(x) for x in line.split()]
                rows.append(row)
            except ValueError:
                print("Invalid input. Please enter numbers separated by spaces.")
        
        if not rows:
            print("No data entered. Creating empty matrix.")
            return cls()
        
        # Check if all rows have same length
        col_len = len(rows[0])
        if not all(len(row) == col_len for row in rows):
            raise ValueError("All rows must have the same number of columns")
        
        return cls(rows)
    
    @classmethod
    def from_list(cls, flat_list, rows, cols):
        """
        Create a matrix from a flat list.
        
        Parameters:
        -----------
        flat_list : list
            A flat list of values
        rows : int
            Number of rows
        cols : int
            Number of columns
            
        Returns:
        --------
        Matrix
            A matrix created from flat list
        """
        return cls(flat_list, rows, cols)
    
    # ==================== Basic Properties ====================
    
    def shape(self):
        """
        Get the shape of the matrix.
        
        Returns:
        --------
        tuple
            (rows, columns)
        """
        return (self.rows, self.cols)
    
    def size(self):
        """
        Get the total number of elements.
        
        Returns:
        --------
        int
            Total number of elements
        """
        return self.rows * self.cols
    
    def flatten(self):
        """
        Flatten the matrix to a 1D list.
        
        Returns:
        --------
        list
            Flattened list of all elements
        """
        return [element for row in self.data for element in row]
    
    def get_element(self, row, col):
        """
        Get a specific element from the matrix.
        
        Parameters:
        -----------
        row : int
            Row index (0-based)
        col : int
            Column index (0-based)
            
        Returns:
        --------
        float
            The element at specified position
        """
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.data[row][col]
        raise IndexError(f"Index ({row}, {col}) out of bounds for matrix of shape {self.shape()}")
    
    def set_element(self, row, col, value):
        """
        Set a specific element in the matrix.
        
        Parameters:
        -----------
        row : int
            Row index (0-based)
        col : int
            Column index (0-based)
        value : float
            Value to set
        """
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.data[row][col] = value
        else:
            raise IndexError(f"Index ({row}, {col}) out of bounds for matrix of shape {self.shape()}")
    
    # ==================== Arithmetic Operations ====================
    
    def __add__(self, other):
        """
        Matrix addition (+).
        
        Parameters:
        -----------
        other : Matrix or scalar
            Matrix or scalar to add
            
        Returns:
        --------
        Matrix
            Result of addition
        """
        if isinstance(other, (int, float)):
            # Scalar addition
            return Matrix([[element + other for element in row] for row in self.data])
        
        if self.shape() != other.shape():
            raise ValueError(f"Shape mismatch: {self.shape()} + {other.shape()}")
        
        result = [[self.data[i][j] + other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result)
    
    def __radd__(self, other):
        """Reverse addition (scalar + matrix)."""
        return self.__add__(other)
    
    def __sub__(self, other):
        """
        Matrix subtraction (-).
        
        Parameters:
        -----------
        other : Matrix or scalar
            Matrix or scalar to subtract
            
        Returns:
        --------
        Matrix
            Result of subtraction
        """
        if isinstance(other, (int, float)):
            # Scalar subtraction
            return Matrix([[element - other for element in row] for row in self.data])
        
        if self.shape() != other.shape():
            raise ValueError(f"Shape mismatch: {self.shape()} - {other.shape()}")
        
        result = [[self.data[i][j] - other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result)
    
    def __rsub__(self, other):
        """Reverse subtraction (scalar - matrix)."""
        if isinstance(other, (int, float)):
            result = [[other - element for element in row] for row in self.data]
            return Matrix(result)
        return NotImplemented
    
    def __mul__(self, other):
        """
        Matrix multiplication (*) - Element-wise or scalar.
        
        Parameters:
        -----------
        other : Matrix or scalar
            Matrix or scalar to multiply
            
        Returns:
        --------
        Matrix
            Result of multiplication
        """
        if isinstance(other, (int, float)):
            # Scalar multiplication
            return Matrix([[element * other for element in row] for row in self.data])
        
        if self.shape() != other.shape():
            raise ValueError(f"Shape mismatch for element-wise multiplication: {self.shape()} * {other.shape()}")
        
        result = [[self.data[i][j] * other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result)
    
    def __rmul__(self, other):
        """Reverse multiplication (scalar * matrix)."""
        return self.__mul__(other)
    
    def __matmul__(self, other):
        """
        Matrix dot product (@) - True matrix multiplication.
        
        Parameters:
        -----------
        other : Matrix
            Matrix to multiply with
            
        Returns:
        --------
        Matrix
            Result of matrix multiplication
        """
        if self.cols != other.rows:
            raise ValueError(f"Shape mismatch for matrix multiplication: {self.shape() @ other.shape()}. "
                           f"Columns of first matrix must equal rows of second matrix.")
        
        result = [[sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                   for j in range(other.cols)] for i in range(self.rows)]
        return Matrix(result)
    
    def __truediv__(self, other):
        """
        Matrix division (/) - Element-wise division.
        
        Parameters:
        -----------
        other : Matrix or scalar
            Matrix or scalar to divide by
            
        Returns:
        --------
        Matrix
            Result of division
        """
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return Matrix([[element / other for element in row] for row in self.data])
        
        if self.shape() != other.shape():
            raise ValueError(f"Shape mismatch: {self.shape()} / {other.shape()}")
        
        # Check for zero division
        if any(element == 0 for row in other.data for element in row):
            raise ZeroDivisionError("Division by zero in matrix")
        
        result = [[self.data[i][j] / other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result)
    
    def __rtruediv__(self, other):
        """Reverse division (scalar / matrix)."""
        if isinstance(other, (int, float)):
            if any(element == 0 for row in self.data for element in row):
                raise ZeroDivisionError("Division by zero in matrix")
            result = [[other / element for element in row] for row in self.data]
            return Matrix(result)
        return NotImplemented
    
    def __mod__(self, other):
        """
        Element-wise modulo operation.
        
        Parameters:
        -----------
        other : Matrix or scalar
            Matrix or scalar for modulo
            
        Returns:
        --------
        Matrix
            Result of modulo operation
        """
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Modulo by zero")
            return Matrix([[element % other for element in row] for row in self.data])
        
        if self.shape() != other.shape():
            raise ValueError(f"Shape mismatch: {self.shape()} % {other.shape()}")
        
        result = [[self.data[i][j] % other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result)
    
    def __pow__(self, other):
        """
        Element-wise power operation.
        
        Parameters:
        -----------
        other : int or float
            Exponent value
            
        Returns:
        --------
        Matrix
            Result of power operation
        """
        if isinstance(other, (int, float)):
            result = [[element ** other for element in row] for row in self.data]
            return Matrix(result)
        return NotImplemented
    
    # ==================== In-place Operations ====================
    
    def __iadd__(self, other):
        """In-place addition."""
        if isinstance(other, (int, float)):
            self.data = [[element + other for element in row] for row in self.data]
        else:
            if self.shape() != other.shape():
                raise ValueError(f"Shape mismatch: {self.shape()} + {other.shape()}")
            self.data = [[self.data[i][j] + other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return self
    
    def __isub__(self, other):
        """In-place subtraction."""
        if isinstance(other, (int, float)):
            self.data = [[element - other for element in row] for row in self.data]
        else:
            if self.shape() != other.shape():
                raise ValueError(f"Shape mismatch: {self.shape()} - {other.shape()}")
            self.data = [[self.data[i][j] - other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return self
    
    def __imul__(self, other):
        """In-place multiplication."""
        if isinstance(other, (int, float)):
            self.data = [[element * other for element in row] for row in self.data]
        else:
            if self.shape() != other.shape():
                raise ValueError(f"Shape mismatch: {self.shape()} * {other.shape()}")
            self.data = [[self.data[i][j] * other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return self
    
    # ==================== Comparison Operations ====================
    
    def __eq__(self, other):
        """Check equality."""
        if not isinstance(other, Matrix):
            return False
        if self.shape() != other.shape():
            return False
        return all(self.data[i][j] == other.data[i][j] for i in range(self.rows) for j in range(self.cols))
    
    def __ne__(self, other):
        """Check inequality."""
        return not self.__eq__(other)
    
    def __lt__(self, other):
        """Element-wise less than."""
        if isinstance(other, (int, float)):
            return Matrix([[element < other for element in row] for row in self.data])
        if self.shape() != other.shape():
            raise ValueError(f"Shape mismatch: {self.shape()} < {other.shape()}")
        return Matrix([[self.data[i][j] < other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])
    
    def __le__(self, other):
        """Element-wise less than or equal."""
        if isinstance(other, (int, float)):
            return Matrix([[element <= other for element in row] for row in self.data])
        if self.shape() != other.shape():
            raise ValueError(f"Shape mismatch: {self.shape()} <= {other.shape()}")
        return Matrix([[self.data[i][j] <= other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])
    
    def __gt__(self, other):
        """Element-wise greater than."""
        if isinstance(other, (int, float)):
            return Matrix([[element > other for element in row] for row in self.data])
        if self.shape() != other.shape():
            raise ValueError(f"Shape mismatch: {self.shape()} > {other.shape()}")
        return Matrix([[self.data[i][j] > other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])
    
    def __ge__(self, other):
        """Element-wise greater than or equal."""
        if isinstance(other, (int, float)):
            return Matrix([[element >= other for element in row] for row in self.data])
        if self.shape() != other.shape():
            raise ValueError(f"Shape mismatch: {self.shape()} >= {other.shape()}")
        return Matrix([[self.data[i][j] >= other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])
    
    # ==================== Matrix Operations ====================
    
    def transpose(self):
        """
        Get the transpose of the matrix.
        
        Returns:
        --------
        Matrix
            Transposed matrix
            
        Examples:
        ---------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m.transpose()
        """
        result = [[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)]
        return Matrix(result)
    
    @property
    def T(self):
        """Property for transpose."""
        return self.transpose()
    
    def trace(self):
        """
        Get the trace of the matrix (sum of diagonal elements).
        
        Returns:
        --------
        float
            Trace of the matrix
        """
        if self.rows != self.cols:
            raise ValueError("Trace is only defined for square matrices")
        return sum(self.data[i][i] for i in range(self.rows))
    
    def determinant(self):
        """
        Calculate the determinant of the matrix.
        
        Returns:
        --------
        float
            Determinant of the matrix
            
        Examples:
        ---------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m.determinant()
        """
        if self.rows != self.cols:
            raise ValueError("Determinant is only defined for square matrices")
        
        n = self.rows
        if n == 1:
            return self.data[0][0]
        if n == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        
        # For larger matrices, use Laplace expansion
        det = 0
        for j in range(n):
            det += ((-1) ** j) * self.data[0][j] * self._minor(0, j).determinant()
        return det
    
    def _minor(self, row, col):
        """Get the minor of a matrix (remove specified row and column)."""
        return Matrix([[self.data[i][j] for j in range(self.cols) if j != col]
                      for i in range(self.rows) if i != row])
    
    def inverse(self):
        """
        Calculate the inverse of the matrix.
        
        Returns:
        --------
        Matrix
            Inverse of the matrix
            
        Examples:
        ---------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m.inverse()
        """
        if self.rows != self.cols:
            raise ValueError("Inverse is only defined for square matrices")
        
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular and cannot be inverted")
        
        n = self.rows
        if n == 1:
            return Matrix([[1 / self.data[0][0]]])
        
        # Calculate cofactor matrix and transpose
        cofactor = [[(((-1) ** (i + j)) * self._minor(i, j).determinant())
                     for j in range(n)] for i in range(n)]
        
        adjugate = Matrix(cofactor).transpose()
        
        # Divide by determinant
        return adjugate * (1 / det)
    
    def dot(self, other):
        """
        Calculate dot product of two matrices.
        
        Parameters:
        -----------
        other : Matrix
            Matrix to calculate dot product with
            
        Returns:
        --------
        Matrix
            Result of dot product
        """
        return self.__matmul__(other)
    
    def cross(self, other):
        """
        Calculate cross product of two matrices.
        
        Parameters:
        -----------
        other : Matrix
            Matrix to calculate cross product with
            
        Returns:
        --------
        Matrix
            Result of cross product
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(f"Shape mismatch: {self.shape()} cross {other.shape()}")
        
        # Element-wise cross product (Hadamard product)
        result = [[self.data[i][j] * other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result)
    
    def hadamard_product(self, other):
        """
        Calculate Hadamard product (element-wise multiplication).
        
        Parameters:
        -----------
        other : Matrix
            Matrix to calculate Hadamard product with
            
        Returns:
        --------
        Matrix
            Result of Hadamard product
        """
        return self.cross(other)
    
    def kronecker_product(self, other):
        """
        Calculate Kronecker product.
        
        Parameters:
        -----------
        other : Matrix
            Matrix to calculate Kronecker product with
            
        Returns:
        --------
        Matrix
            Result of Kronecker product
        """
        result_rows = self.rows * other.rows
        result_cols = self.cols * other.cols
        result = [[0] * result_cols for _ in range(result_rows)]
        
        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(other.rows):
                    for l in range(other.cols):
                        result[i * other.rows + k][j * other.cols + l] = self.data[i][j] * other.data[k][l]
        
        return Matrix(result)
    
    # ==================== Element-wise Functions ====================
    
    def abs(self):
        """Apply absolute value to all elements."""
        return Matrix([[abs(element) for element in row] for row in self.data])
    
    def sqrt(self):
        """Apply square root to all elements."""
        return Matrix([[math.sqrt(element) for element in row] for row in self.data])
    
    def log(self):
        """Apply natural logarithm to all elements."""
        return Matrix([[math.log(element) for element in row] for row in self.data])
    
    def log10(self):
        """Apply base-10 logarithm to all elements."""
        return Matrix([[math.log10(element) for element in row] for row in self.data])
    
    def exp(self):
        """Apply exponential to all elements."""
        return Matrix([[math.exp(element) for element in row] for row in self.data])
    
    def sin(self):
        """Apply sine to all elements (in radians)."""
        return Matrix([[math.sin(element) for element in row] for row in self.data])
    
    def cos(self):
        """Apply cosine to all elements (in radians)."""
        return Matrix([[math.cos(element) for element in row] for row in self.data])
    
    def tan(self):
        """Apply tangent to all elements (in radians)."""
        return Matrix([[math.tan(element) for element in row] for row in self.data])
    
    def floor(self):
        """Apply floor to all elements."""
        return Matrix([[math.floor(element) for element in row] for row in self.data])
    
    def ceil(self):
        """Apply ceiling to all elements."""
        return Matrix([[math.ceil(element) for element in row] for row in self.data])
    
    def round(self, decimals=0):
        """Round all elements."""
        if decimals == 0:
            return Matrix([[round(element) for element in row] for row in self.data])
        return Matrix([[round(element, decimals) for element in row] for row in self.data])
    
    # ==================== Aggregate Functions ====================
    
    def sum(self):
        """
        Calculate sum of all elements.
        
        Returns:
        --------
        float
            Sum of all elements
        """
        return sum(sum(row) for row in self.data)
    
    def product(self):
        """
        Calculate product of all elements.
        
        Returns:
        --------
        float
            Product of all elements
        """
        result = 1
        for row in self.data:
            for element in row:
                result *= element
        return result
    
    def min(self):
        """
        Find minimum element.
        
        Returns:
        --------
        float
            Minimum element
        """
        return min(min(row) for row in self.data)
    
    def max(self):
        """
        Find maximum element.
        
        Returns:
        --------
        float
            Maximum element
        """
        return max(max(row) for row in self.data)
    
    def mean(self):
        """
        Calculate mean of all elements.
        
        Returns:
        --------
        float
            Mean of all elements
        """
        return self.sum() / self.size()
    
    def std(self):
        """
        Calculate standard deviation.
        
        Returns:
        --------
        float
            Standard deviation
        """
        m = self.mean()
        variance = sum((element - m) ** 2 for row in self.data for element in row) / self.size()
        return math.sqrt(variance)
    
    def variance(self):
        """
        Calculate variance.
        
        Returns:
        --------
        float
            Variance
        """
        m = self.mean()
        return sum((element - m) ** 2 for row in self.data for element in row) / self.size()
    
    # ==================== Row/Column Operations ====================
    
    def get_row(self, index):
        """Get a specific row as a Matrix."""
        if 0 <= index < self.rows:
            return Matrix([self.data[index][:]])
        raise IndexError(f"Row index {index} out of bounds")
    
    def get_col(self, index):
        """Get a specific column as a Matrix."""
        if 0 <= index < self.cols:
            return Matrix([[self.data[i][index]] for i in range(self.rows)])
        raise IndexError(f"Column index {index} out of bounds")
    
    def row_sum(self):
        """Sum of each row."""
        return Matrix([[sum(row) for row in self.data]])
    
    def col_sum(self):
        """Sum of each column."""
        return Matrix([[sum(self.data[i][j] for i in range(self.rows))] for j in range(self.cols)])
    
    def row_mean(self):
        """Mean of each row."""
        return Matrix([[sum(row) / self.cols] for row in self.data])
    
    def col_mean(self):
        """Mean of each column."""
        return Matrix([[sum(self.data[i][j] for i in range(self.rows)) / self.rows] for j in range(self.cols)])
    
    # ==================== Matrix Transformations ====================
    
    def reshape(self, new_rows, new_cols):
        """
        Reshape the matrix.
        
        Parameters:
        -----------
        new_rows : int
            New number of rows
        new_cols : int
            New number of columns
            
        Returns:
        --------
        Matrix
            Reshaped matrix
        """
        if new_rows * new_cols != self.size():
            raise ValueError(f"Cannot reshape from {self.shape()} to ({new_rows}, {new_cols})")
        
        flat = self.flatten()
        return Matrix(flat, new_rows, new_cols)
    
    def flatten_matrix(self):
        """Flatten matrix to 1D."""
        return Matrix([self.flatten()], 1, self.size())
    
    def append(self, other, axis=0):
        """
        Append another matrix.
        
        Parameters:
        -----------
        other : Matrix
            Matrix to append
        axis : int, optional
            0 for row-wise, 1 for column-wise (default: 0)
            
        Returns:
        --------
        Matrix
            Appended matrix
        """
        if axis == 0:
            # Row-wise
            if self.cols != other.cols:
                raise ValueError("Columns must match for row-wise append")
            return Matrix(self.data + other.data)
        else:
            # Column-wise
            if self.rows != other.rows:
                raise ValueError("Rows must match for column-wise append")
            result = [self.data[i] + other.data[i] for i in range(self.rows)]
            return Matrix(result)
    
    # ==================== Utility Methods ====================
    
    def copy(self):
        """Create a deep copy of the matrix."""
        return Matrix([row[:] for row in self.data])
    
    def apply_function(self, func):
        """
        Apply a function to each element.
        
        Parameters:
        -----------
        func : callable
            Function to apply to each element
            
        Returns:
        --------
        Matrix
            Matrix with function applied
        """
        return Matrix([[func(element) for element in row] for row in self.data])
    
    # ==================== String Representation ====================
    
    def __str__(self):
        """String representation of the matrix."""
        if self.rows == 0:
            return "Empty Matrix"
        
        # Calculate column widths
        col_widths = []
        for j in range(self.cols):
            max_width = max(len(f"{self.data[i][j]:.4f}") for i in range(self.rows))
            col_widths.append(max_width)
        
        # Format each row
        lines = []
        for i in range(self.rows):
            row_str = "  ".join(f"{self.data[i][j]:{col_widths[j]}.4f}" for j in range(self.cols))
            lines.append(f"[ {row_str} ]")
        
        return "\n".join(lines)
    
    def __repr__(self):
        """Detailed representation of the matrix."""
        return f"Matrix({self.data}, rows={self.rows}, cols={self.cols})"
    
    def display(self, title=None):
        """
        Display the matrix with optional title.
        
        Parameters:
        -----------
        title : str, optional
            Title to display
        """
        if title:
            print(f"\n{'=' * 40}")
            print(f"  {title}")
            print('=' * 40)
        print(self)
        print()


# ==================== Standalone Functions ====================

def add(a, b):
    """Add two matrices or a matrix and a scalar."""
    return a + b

def subtract(a, b):
    """Subtract two matrices or a matrix and a scalar."""
    return a - b

def multiply(a, b):
    """Multiply two matrices or a matrix and a scalar."""
    return a * b

def divide(a, b):
    """Divide a matrix by another matrix or a scalar."""
    return a / b

def matmul(a, b):
    """Matrix multiplication (dot product)."""
    return a @ b

def transpose(matrix):
    """Transpose a matrix."""
    return matrix.transpose()

def determinant(matrix):
    """Calculate determinant of a matrix."""
    return matrix.determinant()

def inverse(matrix):
    """Calculate inverse of a matrix."""
    return matrix.inverse()

def zeros(rows, cols):
    """Create a zero matrix."""
    return Matrix.zeros(rows, cols)

def ones(rows, cols):
    """Create a matrix of ones."""
    return Matrix.ones(rows, cols)

def identity(size):
    """Create an identity matrix."""
    return Matrix.identity(size)

def random_matrix(rows, cols, min_val=0, max_val=1):
    """Create a random matrix."""
    return Matrix.random(rows, cols, min_val, max_val)

def random_int_matrix(rows, cols, min_val=0, max_val=10):
    """Create a random integer matrix."""
    return Matrix.random_int(rows, cols, min_val, max_val)


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    # Interactive demo when run directly
    print("=" * 60)
    print("  Welcome to MyNumpy - Custom Matrix Library!")
    print("=" * 60)
    print("\nThis library provides numpy-like functionality for matrices.")
    print("\nAvailable features:")
    print("  - Matrix creation: zeros, ones, identity, random, diagonal")
    print("  - Arithmetic: +, -, *, /, @, %, **")
    print("  - Matrix operations: transpose, inverse, determinant")
    print("  - Math functions: sin, cos, tan, exp, log, sqrt")
    print("  - Aggregates: sum, min, max, mean, std, variance")
    print("  - And much more!")
    print("\n" + "=" * 60)
    
    # Create a sample matrix
    print("\n--- Sample Matrix Operations ---\n")
    
    # Create matrices
    A = Matrix([[1, 2, 3], 
                [4, 5, 6], 
                [7, 8, 9]])
    print("Matrix A:")
    A.display("Matrix A (3x3)")
    
    B = Matrix([[9, 8, 7],
                [6, 5, 4],
                [3, 2, 1]])
    print("Matrix B:")
    B.display("Matrix B (3x3)")
    
    # Addition
    C = A + B
    C.display("A + B (Addition)")
    
    # Multiplication (element-wise)
    D = A * B
    D.display("A * B (Element-wise Multiplication)")
    
    # Matrix multiplication
    E = A @ B
    E.display("A @ B (Matrix Multiplication)")
    
    # Transpose
    F = A.T
    F.display("A.T (Transpose)")
    
    # Determinant
    det = A.determinant()
    print(f"Determinant of A: {det}")
    
    # Inverse of a 2x2 matrix
    G = Matrix([[4, 7], 
                [2, 6]])
    G_inv = G.inverse()
    G.display("Matrix G (2x2)")
    G_inv.display("Inverse of G")
    
    # Verify inverse
    identity_2x2 = G @ G_inv
    identity_2x2.display("G @ G_inv (Should be Identity)")
    
    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)
