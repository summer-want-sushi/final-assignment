# tools/math_tools.py
import math
from typing import Union, List, Any
from .utils import logger

Number = Union[int, float]

class MathTools:
    """Simple math tools for basic calculations and utilities"""
    
    @staticmethod
    def add(a: Number, b: Number) -> Number:
        """Return the sum of a and b"""
        return a + b
    
    @staticmethod
    def subtract(a: Number, b: Number) -> Number:
        """Return the difference of a and b"""
        return a - b
    
    @staticmethod
    def multiply(a: Number, b: Number) -> Number:
        """Return the product of a and b"""
        return a * b
    
    @staticmethod
    def divide(a: Number, b: Number) -> Union[Number, str]:
        """Return the division of a by b, handle division by zero"""
        if b == 0:
            return 'Error: Division by zero'
        return a / b
    
    @staticmethod
    def power(base: Number, exponent: Number) -> Number:
        """Return base raised to the power of exponent"""
        return base ** exponent
    
    @staticmethod
    def factorial(n: int) -> Union[int, str]:
        """Return factorial of n (non-negative integer)"""
        if not isinstance(n, int) or n < 0:
            return 'Error: Input must be a non-negative integer'
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
    
    @staticmethod
    def square_root(n: Number) -> Union[float, str]:
        """Return square root of n"""
        if n < 0:
            return 'Error: Cannot calculate square root of negative number'
        return math.sqrt(n)
    
    @staticmethod
    def percentage(part: Number, whole: Number) -> Union[float, str]:
        """Calculate percentage: (part/whole) * 100"""
        if whole == 0:
            return 'Error: Cannot calculate percentage with zero denominator'
        return (part / whole) * 100
    
    @staticmethod
    def average(numbers: List[Number]) -> Union[float, str]:
        """Calculate average of a list of numbers"""
        if not numbers:
            return 'Error: Cannot calculate average of empty list'
        return sum(numbers) / len(numbers)
    
    @staticmethod
    def round_number(n: Number, decimals: int = 2) -> Number:
        """Round number to specified decimal places"""
        return round(n, decimals)
    
    @staticmethod
    def absolute(n: Number) -> Number:
        """Return absolute value of n"""
        return abs(n)
    
    @staticmethod
    def min_value(numbers: List[Number]) -> Union[Number, str]:
        """Find minimum value in list"""
        if not numbers:
            return 'Error: Cannot find minimum of empty list'
        return min(numbers)
    
    @staticmethod
    def max_value(numbers: List[Number]) -> Union[Number, str]:
        """Find maximum value in list"""
        if not numbers:
            return 'Error: Cannot find maximum of empty list'
        return max(numbers)
    
    @staticmethod
    def calculate_compound_interest(principal: Number, rate: Number, time: Number, compounds_per_year: int = 1) -> float:
        """
        Calculate compound interest
        Formula: A = P(1 + r/n)^(nt)
        """
        return principal * (1 + rate/compounds_per_year) ** (compounds_per_year * time)
    
    @staticmethod
    def solve_quadratic(a: Number, b: Number, c: Number) -> Union[tuple, str]:
        """
        Solve quadratic equation ax² + bx + c = 0
        Returns tuple of solutions or error message
        """
        if a == 0:
            return 'Error: Not a quadratic equation (a cannot be 0)'
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return 'Error: No real solutions (negative discriminant)'
        elif discriminant == 0:
            solution = -b / (2*a)
            return (solution, solution)
        else:
            sqrt_discriminant = math.sqrt(discriminant)
            solution1 = (-b + sqrt_discriminant) / (2*a)
            solution2 = (-b - sqrt_discriminant) / (2*a)
            return (solution1, solution2)

# Convenience functions for direct use
def add(a: Number, b: Number) -> Number:
    """Add two numbers"""
    return MathTools.add(a, b)

def subtract(a: Number, b: Number) -> Number:
    """Subtract two numbers"""
    return MathTools.subtract(a, b)

def multiply(a: Number, b: Number) -> Number:
    """Multiply two numbers"""
    return MathTools.multiply(a, b)

def divide(a: Number, b: Number) -> Union[Number, str]:
    """Divide two numbers"""
    return MathTools.divide(a, b)

def power(base: Number, exponent: Number) -> Number:
    """Raise base to power of exponent"""
    return MathTools.power(base, exponent)

def factorial(n: int) -> Union[int, str]:
    """Calculate factorial of n"""
    return MathTools.factorial(n)

def square_root(n: Number) -> Union[float, str]:
    """Calculate square root"""
    return MathTools.square_root(n)

def percentage(part: Number, whole: Number) -> Union[float, str]:
    """Calculate percentage"""
    return MathTools.percentage(part, whole)

def average(numbers: List[Number]) -> Union[float, str]:
    """Calculate average of numbers"""
    return MathTools.average(numbers)

def calculate_expression(expression: str) -> Union[Number, str]:
    """
    Safely evaluate mathematical expressions
    WARNING: Only use with trusted input
    """
    try:
        # Only allow safe mathematical operations
        allowed_chars = set('0123456789+-*/().^ ')
        if not all(c in allowed_chars for c in expression.replace('**', '^')):
            return 'Error: Invalid characters in expression'
        
        # Replace ^ with ** for Python exponentiation
        safe_expression = expression.replace('^', '**')
        
        # Evaluate the expression
        result = eval(safe_expression)
        return result
    except ZeroDivisionError:
        return 'Error: Division by zero in expression'
    except Exception as e:
        return f'Error: Invalid expression - {str(e)}'

# Example usage and testing
if __name__ == "__main__":
    # Test basic operations
    print("Basic Operations:")
    print(f"5 + 3 = {add(5, 3)}")
    print(f"10 - 4 = {subtract(10, 4)}")
    print(f"6 * 7 = {multiply(6, 7)}")
    print(f"15 / 3 = {divide(15, 3)}")
    print(f"2^8 = {power(2, 8)}")
    
    print("\nAdvanced Operations:")
    print(f"√16 = {square_root(16)}")
    print(f"5! = {factorial(5)}")
    print(f"Average of [1,2,3,4,5] = {average([1,2,3,4,5])}")
    percent_result = percentage(25, 100)
    if isinstance(percent_result, float):
        print(f"25% of 200 = {percent_result * 200 / 100}")
    else:
        print(f"25% of 200 = {percent_result}")
    
    print("\nQuadratic Equation (x² - 5x + 6 = 0):")
    solutions = MathTools.solve_quadratic(1, -5, 6)
    print(f"Solutions: {solutions}")
