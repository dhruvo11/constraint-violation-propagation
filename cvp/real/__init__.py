"""Real documents: readers for public datasets and the rules that apply to them.

The synthetic document types (cvp/doctypes) render their own pages and know every field's
position. Real documents come with whatever their dataset provides, such as ground-truth
values and word boxes. This package turns them into the same fields-and-constraints form, so
the constraint branch can be measured on genuine documents and on forgeries of them.
"""
