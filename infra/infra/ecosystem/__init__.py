DOMAIN = 'mictlan.site'


# 遅延ロードしないと循環参照になる
def gen_compose():
    from .compose import gen_compose
    gen_compose()
