# __init__.py 不是必须“写内容”
# 1) 让目录成为“包”
# 2) 包被导入时会执行 __init__.py
# 3) 可以在 __init__.py 中定义包的公共接口

# .是相对导入，表示从当前包中导入，若全在根目录下就可以不用.

"""
    The overall process of the experiment：
        Input sequence
        [source][sep][noise][query]
                ↓
        Windowed Transformer Encoder(natural / forced)
                ↓
        Hidden states h
                ↓
        Write Policy
        source-only / prefix-all / source-pinned / intermediate
                ↓
        External KV Buffer
        keys / values / labels
                ↓
        Memory Read at query positions
        naive read / gated read
                ↓
        Prediction
        copy source tokens
"""