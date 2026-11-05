"""Qwen-Image-Edit schnell: bei Stärke 1 rechnet je Schritt nur der erste Aufruf des Bildteils (der Gegendurchgang
ist wirkungslos), Paare stimmen nach neu(), Attribute gehen durch."""
from bildwerk.modelle.qwen_image_edit_schnell import ANGABEN, OhneGegendurchgang


class Bildteil:
    def __init__(self) -> None:
        self.aufrufe, self.gewicht = 0, 7

    def __call__(self, x: int) -> int:
        self.aufrufe += 1
        return x * 10


def test_rechnet_nur_den_ersten_aufruf_je_schritt():
    b = Bildteil()
    o = OhneGegendurchgang(b)
    assert [o(1), o(2), o(3), o(4)] == [10, 10, 30, 30]   # zweiter Aufruf je Schritt: das erste Ergebnis
    assert b.aufrufe == 2 and o.gewicht == 7
    o(5)                                                  # ein abgebrochener Schritt ...
    o.neu()                                               # ... das nächste Bild beginnt wieder beim Paar
    assert [o(6), o(7)] == [60, 60] and b.aufrufe == 4


def test_nur_bei_staerke_eins():
    assert ANGABEN.guidance == 1.0 and ANGABEN.schritte.vorgabe == 8
