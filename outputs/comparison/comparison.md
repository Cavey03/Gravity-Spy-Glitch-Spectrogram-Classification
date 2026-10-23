# Gravity Spy -- model comparison

## Headline metrics

| Model | macro-F1 | Accuracy | Balanced acc. | weighted-F1 | macro-P | macro-R | Params | Epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `transformer` | 0.9421 | 0.9712 | 0.9346 | 0.9712 | 0.9618 | 0.9346 | 27,536,272 | 20 |
| `custom_cnn` | 0.9332 | 0.9686 | 0.9267 | 0.9684 | 0.9511 | 0.9267 | 2,359,542 | 57 |
| `transfer_cnn` | 0.9323 | 0.9686 | 0.9248 | 0.9686 | 0.9498 | 0.9248 | 20,205,670 | 25 |

*Evaluated on the shared **test** split, sorted by macro-F1.*

**Read macro-F1, not accuracy.** The class distribution is skewed enough that accuracy stays around 97% regardless of how the rare classes are handled.

## Per-class F1

| Class | Support | `transformer` F1 | `custom_cnn` F1 | `transfer_cnn` F1 |
|---|---:|---:|---:|---:|
| Blip | 272 | 0.993 | 0.991 | 0.985 |
| Koi_Fish | 102 | 0.990 | 0.990 | 0.980 |
| Low_Frequency_Burst | 90 | 0.983 | 0.949 | 0.966 |
| Light_Modulation | 78 | 0.974 | 0.974 | 0.969 |
| Power_Line | 68 | 0.993 | 0.993 | 0.986 |
| Scattered_Light | 67 | 0.962 | 0.977 | 0.970 |
| Low_Frequency_Lines | 66 | 0.921 | 0.908 | 0.955 |
| Extremely_Loud | 64 | 0.961 | 0.969 | 0.955 |
| Violin_Mode | 64 | 0.992 | 0.985 | 0.984 |
| 1080Lines | 50 | 0.969 | 0.969 | 0.980 |
| Scratchy | 50 | 0.952 | 0.980 | 0.961 |
| Whistle | 45 | 0.989 | 0.967 | 0.966 |
| Helix | 42 | 1.000 | 1.000 | 1.000 |
| Repeating_Blips | 37 | 0.946 | 0.959 | 0.973 |
| No_Glitch | 21 | 0.750 | 0.818 | 0.878 |
| Tomte | 13 | 1.000 | 0.963 | 0.960 |
| None_of_the_Above | 11 | 0.842 | 0.750 | 0.720 |
| Chirp | 10 | 1.000 | 1.000 | 1.000 |
| 1400Ripples | 9 | 0.900 | 0.947 | 0.947 |
| Air_Compressor | 9 | 0.941 | 0.941 | 0.941 |
| Wandering_Line | 7 | 1.000 | 0.833 | 0.769 |
| Paired_Doves | 4 | 0.667 | 0.667 | 0.667 |
