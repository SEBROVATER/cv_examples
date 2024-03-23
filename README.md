# cv_examples
Examples of some self made CV algorithms.

---
## Cards suits recognition:

![heart.gif](cards%2Fheart.gif) ![diamond.gif](cards%2Fdiamond.gif)

```Python
def heart_or_diamond(suit_binary: npt.NDArray[np.uint8]) -> str:
    belt_y = np.count_nonzero(suit_binary, axis=1).argmax()
    belt_x = np.count_nonzero(suit_binary, axis=0).argmax()

    nonzeros = np.nonzero(suit_binary)
    y1, x1 = np.minimum.reduce(nonzeros, axis=1)
    y2, x2 = np.maximum.reduce(nonzeros, axis=1)

    yc = (y1 + y2) / 2
    diff_y = abs(yc - belt_y)
    gap_y = diff_y / (y2 - y1)

    xc = (x1 + x2) / 2
    diff_x = abs(xc - belt_x)
    gap_x = diff_x / (x2 - x1)

    top_belt_y = suit_binary[:, belt_x].argmax()

    if gap_y < 0.15 and gap_x < 0.15 and np.count_nonzero(suit_binary[:top_belt_y]) <= 1:
        return "d"
    return "h"
```
```Python
def club_or_spade(suit_binary: npt.NDArray[np.uint8]) -> str:
    belt_y = np.count_nonzero(suit_binary, axis=1).argmax()
    top_half = suit_binary[: belt_y + 1]

    nonzeros = np.nonzero(top_half)
    y1, x1 = np.minimum.reduce(nonzeros, axis=1)
    y2, x2 = np.maximum.reduce(nonzeros, axis=1)

    gap = int((x2 - x1) / 4)
    center = top_half[int((y1 + y2) / 2) + 1:, x1 + gap: x2 - gap + 1]

    if center.all():
        return "s"
    else:
        return "c"
```
