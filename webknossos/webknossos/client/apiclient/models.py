import attr


@attr.s(auto_attribs=True)
class ApiShortLink:
    longLink: str
