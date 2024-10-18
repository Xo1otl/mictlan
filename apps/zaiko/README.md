# Zaiko

在庫管理システム

## Memo

- リモート`35.78.179.182`
- `CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build`でビルド

### schema

zaiko.stock.commands-Added-value
```json
{
    "type": "record",
    "name": "Added",
    "fields": [
        {
            "name": "Sub",
            "type": "string"
        },
        {
            "name": "Name",
            "type": "string"
        },
        {
            "name": "Amount",
            "type": "int"
        }
    ]
}
```

zaiko.stock.commands-ClearedAll-value
```json
{
    "type": "record",
    "name": "ClearedAll",
    "fields": [
        {
            "name": "Sub",
            "type": "string"
        }
    ]
}
```

zaiko.stock.commands-Sold-value
```json
{
    "type": "record",
    "name": "Sold",
    "fields": [
        {
            "name": "Sub",
            "type": "string"
        },
        {
            "name": "Name",
            "type": "string"
        },
        {
            "name": "Amount",
            "type": "int"
        },
        {
            "name": "Price",
            "type": "string"
        }
    ]
}
```

zaiko.stock.commands-key
```json
{
    "name": "sub",
    "type": "string"
}
```

zaiko.stock.projections-AggregateUpdated-value
```json
{
    "type": "record",
    "name": "AggregateUpdated",
    "fields": [
        {
            "name": "Stocks",
            "type": {
                "type": "map",
                "values": "int"
            }
        },
        {
            "name": "Sales",
            "type": "string"
        }
    ]
}
```

zaiko.stock.projections-key
```json
{
    "name": "Sub",
    "type": "string"
}
```