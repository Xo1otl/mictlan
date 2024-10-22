def maxUniqueSplit(s: str) -> int:
    # セットでユニークなサブストリングを保持
    unique_substrings = set()
    
    def backtrack(start, count):
        nonlocal max_count
        if start == len(s):
            max_count = max(max_count, count)
            return
        
        for end in range(start + 1, len(s) + 1):
            substr = s[start:end]
            if substr not in unique_substrings:
                unique_substrings.add(substr)
                backtrack(end, count + 1)
                unique_substrings.remove(substr)  # backtracking
    
    max_count = 0
    backtrack(0, 0)
    return max_count

# テストケース
print(maxUniqueSplit("ababccc"))  # Output: 5
print(maxUniqueSplit("aba"))      # Output: 2
print(maxUniqueSplit("aa"))       # Output: 1