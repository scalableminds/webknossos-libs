# Logged User Times

This example uses the [`User` class](../../api/webknossos/administration/user.md#webknossos.administration.User) to load the logged times
for all users of whom the current user is admin or team-manager.

*This example additionally needs the pandas and tabulate packages.*

```python
--8<--
webknossos/examples/user_times.py
--8<--
```

This results in an output similar to

```
Logged User Times 2021:

| email                  |    1 |    2 |    3 |    4 |   5 |   6 |   7 |
|:-----------------------|-----:|-----:|-----:|-----:|----:|----:|----:|
| abc@mail.com           |    0 |    0 |    0 |    0 |   0 |   0 |   0 |
| somebody@mail.com      |    0 |    0 |   16 |  210 |   0 | 271 | 150 |
| someone@mail.com       |    0 |    0 |  553 |    0 |   0 |   0 |   0 |
| taylor.tester@mail.com |    0 |    0 |    0 | 1746 |   0 |   0 | 486 |
| tony.test@mail.com     |   36 |    0 |  158 |    0 |  20 |   0 | 452 |
| xyz@mail.com           |    0 |  260 |  674 |  903 |   0 | 541 |   0 |
```
