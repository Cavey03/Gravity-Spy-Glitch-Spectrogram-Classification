# Gravity Spy -- model comparison

## Headline metrics

| Model | macro-F1 | Accuracy | Balanced acc. | weighted-F1 | macro-P | macro-R | Params | Epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `custom_cnn` | 0.9332 | 0.9686 | 0.9267 | 0.9684 | 0.9511 | 0.9267 | 2,359,542 | 57 |

*Evaluated on the shared **test** split, sorted by macro-F1.*

**Read macro-F1, not accuracy.** The class distribution is skewed enough that accuracy stays around 97% regardless of how the rare classes are handled.

## Per-class F1

| Class | Support | `custom_cnn` F1 |
|---|---:|---:|
| Blip | 272 | 0.991 |
| Koi_Fish | 102 | 0.990 |
| Low_Frequency_Burst | 90 | 0.949 |
| Light_Modulation | 78 | 0.974 |
| Power_Line | 68 | 0.993 |
| Scattered_Light | 67 | 0.977 |
| Low_Frequency_Lines | 66 | 0.908 |
| Extremely_Loud | 64 | 0.969 |
| Violin_Mode | 64 | 0.985 |
| 1080Lines | 50 | 0.969 |
| Scratchy | 50 | 0.980 |
| Whistle | 45 | 0.967 |
| Helix | 42 | 1.000 |
| Repeating_Blips | 37 | 0.959 |
| No_Glitch | 21 | 0.818 |
| Tomte | 13 | 0.963 |
| None_of_the_Above | 11 | 0.750 |
| Chirp | 10 | 1.000 |
| 1400Ripples | 9 | 0.947 |
| Air_Compressor | 9 | 0.941 |
| Wandering_Line | 7 | 0.833 |
| Paired_Doves | 4 | 0.667 |
