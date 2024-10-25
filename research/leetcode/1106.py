from typing import List


class Solution:
    def parseBoolExpr(self, expression: str) -> bool:
        def parse_args(expr_arg: str) -> List[str]:
            """
            expr_arg: 引数取る演算のかっこの中身の文字列 e.g., "!(&(f)),&(t),|(f,f,t)"
            return: subExprのリスト
            """
            args = []
            nest = 0
            i = 0
            while True:
                i += 1
                # 最後の引数を取り出す
                if i >= len(expr_arg):
                    args.append(expr_arg)
                    break
                if expr_arg[i] == '(':
                    nest += 1
                if expr_arg[i] == ')':
                    nest -= 1
                if nest == 0 and expr_arg[i] == ',':
                    args.append(expr_arg[:i])
                    expr_arg = expr_arg[i + 1:]
                    i = 0
            return args

        def parse(expr: str) -> bool:
            if expr[0] == 't':
                return True
            if expr[0] == 'f':
                return False
            if expr[0] == '!':
                return not parse(expr=expr[2:-1])
            if expr[0] == '&':
                return all(parse(arg) for arg in parse_args(expr[2:-1]))
            if expr[0] == '|':
                return any(parse(arg) for arg in parse_args(expr[2:-1]))
            exit(1)

        return parse(expression)


result = Solution().parseBoolExpr(expression="&(|(f))")
print(result)
