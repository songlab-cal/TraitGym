import polars as pl
import pytest

from traitgym.intervals import add_exon, add_tss, get_exon, get_tss, load_annotation


class TestLoadAnnotation:
    def test_loads_gtf(self) -> None:
        ann = load_annotation("other/results/annotation.gtf.gz")
        assert ann.shape[0] > 0
        assert set(ann.columns) == {
            "chrom",
            "source",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "frame",
            "attribute",
        }

    def test_converts_to_0_based(self) -> None:
        ann = load_annotation("other/results/annotation.gtf.gz")
        # GTF is 1-based, BED is 0-based, so start should be decremented
        # First gene on chr1 starts at 11869 in GTF, should be 11868 in BED
        chr1 = ann.filter(pl.col("chrom") == "1")
        assert chr1["start"].min() == 11868


class TestGetTss:
    @pytest.fixture
    def annotation(self) -> pl.DataFrame:
        return load_annotation("other/results/annotation.gtf.gz")

    def test_output_schema(self, annotation: pl.DataFrame) -> None:
        tss = get_tss(annotation)
        assert tss.columns == ["chrom", "start", "end", "gene_id"]

    def test_tss_is_1bp(self, annotation: pl.DataFrame) -> None:
        tss = get_tss(annotation)
        assert (tss["end"] - tss["start"] == 1).all()

    def test_deduplicates(self, annotation: pl.DataFrame) -> None:
        tss = get_tss(annotation)
        assert tss.shape[0] == tss.unique().shape[0]

    def test_sorted(self, annotation: pl.DataFrame) -> None:
        tss = get_tss(annotation)
        sorted_tss = tss.sort(["chrom", "start", "end"])
        assert tss.equals(sorted_tss)

    def test_matches_original_implementation(self, annotation: pl.DataFrame) -> None:
        from gpn.data import load_table

        my_tss = get_tss(annotation)

        # Original logic (without unique/sort)
        ann_pd = load_table("other/results/annotation.gtf.gz")
        tx = ann_pd.query('feature=="transcript"').copy()
        tx["gene_id"] = tx.attribute.str.extract(r'gene_id "([^;]*)";')
        tx["transcript_biotype"] = tx.attribute.str.extract(
            r'transcript_biotype "([^;]*)";'
        )
        tx = tx[tx.transcript_biotype == "protein_coding"]
        tss = tx.copy()
        tss["start"] = tss.apply(
            lambda w: w.start if w.strand == "+" else w.end - 1, axis=1
        )
        tss["end"] = tss["start"] + 1
        tss = tss[["chrom", "start", "end", "gene_id"]]
        orig_tss = pl.from_pandas(tss).unique().sort(["chrom", "start", "end", "gene_id"])

        my_sorted = my_tss.sort(["chrom", "start", "end", "gene_id"])
        assert my_sorted.equals(orig_tss)


class TestGetExon:
    @pytest.fixture
    def annotation(self) -> pl.DataFrame:
        return load_annotation("other/results/annotation.gtf.gz")

    def test_output_schema(self, annotation: pl.DataFrame) -> None:
        exon = get_exon(annotation)
        assert exon.columns == ["chrom", "start", "end", "gene_id"]

    def test_filters_protein_coding(self, annotation: pl.DataFrame) -> None:
        exon = get_exon(annotation)
        # All exons should be from protein-coding transcripts
        # Check that we have gene IDs (not null)
        assert exon["gene_id"].null_count() == 0

    def test_filters_canonical_chroms(self, annotation: pl.DataFrame) -> None:
        exon = get_exon(annotation)
        chroms = exon["chrom"].unique().to_list()
        expected_chroms = [str(i) for i in range(1, 23)] + ["X", "Y"]
        assert all(c in expected_chroms for c in chroms)

    def test_deduplicates(self, annotation: pl.DataFrame) -> None:
        exon = get_exon(annotation)
        # Should have no duplicate rows
        assert exon.shape[0] == exon.unique().shape[0]

    def test_sorted(self, annotation: pl.DataFrame) -> None:
        exon = get_exon(annotation)
        sorted_exon = exon.sort(["chrom", "start", "end"])
        assert exon.equals(sorted_exon)

    def test_matches_original_implementation(self, annotation: pl.DataFrame) -> None:
        """Verify output matches the original pandas implementation."""
        from gpn.data import load_table

        from traitgym.variants import CHROMS

        my_exon = get_exon(annotation)

        # Original logic
        ann_pd = load_table("other/results/annotation.gtf.gz")
        exon = ann_pd.query('feature=="exon"').copy()
        exon["gene_id"] = exon.attribute.str.extract(r'gene_id "([^;]*)";')
        exon["transcript_biotype"] = exon.attribute.str.extract(
            r'transcript_biotype "([^;]*)";'
        )
        exon = exon[exon.transcript_biotype == "protein_coding"]
        exon = exon[["chrom", "start", "end", "gene_id"]].drop_duplicates()
        exon = exon[exon.chrom.isin(CHROMS)]
        exon = exon.sort_values(["chrom", "start", "end"])
        orig_exon = pl.from_pandas(exon)

        # Compare sorted by all columns
        my_sorted = my_exon.sort(["chrom", "start", "end", "gene_id"])
        orig_sorted = orig_exon.sort(["chrom", "start", "end", "gene_id"])
        assert my_sorted.equals(orig_sorted)


class TestAddExon:
    @pytest.fixture
    def exon(self) -> pl.DataFrame:
        ann = load_annotation("other/results/annotation.gtf.gz")
        return get_exon(ann)

    @pytest.fixture
    def variants(self) -> pl.DataFrame:
        return pl.DataFrame({
            "chrom": ["1", "1", "1"],
            "pos": [65483, 65583, 200000],  # 50bp from exon, 150bp from exon, far
            "ref": ["A", "A", "A"],
            "alt": ["T", "T", "T"],
            "consequence": ["intron_variant", "intron_variant", "intergenic_variant"],
            "consequence_cre": ["intron_variant", "intron_variant", "intergenic_variant"],
        })

    def test_output_columns(self, variants: pl.DataFrame, exon: pl.DataFrame) -> None:
        result = add_exon(variants, exon)
        assert "exon_dist" in result.columns
        assert "exon_closest_gene_id" in result.columns
        assert "consequence_final" in result.columns

    def test_exon_proximal_within_threshold(self, variants: pl.DataFrame, exon: pl.DataFrame) -> None:
        result = add_exon(variants, exon, exon_proximal_dist=100)
        # First variant is intron_variant within 100bp of exon
        assert result.filter(pl.col("pos") == 65483)["consequence_final"][0] == "exon_proximal"

    def test_exon_proximal_outside_threshold(self, variants: pl.DataFrame, exon: pl.DataFrame) -> None:
        result = add_exon(variants, exon, exon_proximal_dist=10)
        # First variant is intron_variant but >10bp from exon
        assert result.filter(pl.col("pos") == 65483)["consequence_final"][0] == "intron_variant"

    def test_non_intron_not_modified(self, variants: pl.DataFrame, exon: pl.DataFrame) -> None:
        result = add_exon(variants, exon, exon_proximal_dist=1000000)
        # intergenic_variant should not become exon_proximal
        assert result.filter(pl.col("pos") == 200000)["consequence_final"][0] == "intergenic_variant"

    def test_distances_match_bioframe(self, variants: pl.DataFrame, exon: pl.DataFrame) -> None:
        import bioframe as bf

        result = add_exon(variants, exon)

        # Compare with bioframe
        V_pd = variants.to_pandas()
        exon_pd = exon.to_pandas()
        V_pd["start"] = V_pd.pos - 1
        V_pd["end"] = V_pd.pos
        result_bf = bf.closest(V_pd, exon_pd)

        assert result["exon_dist"].to_list() == result_bf["distance"].tolist()


class TestAddTss:
    @pytest.fixture
    def tss(self) -> pl.DataFrame:
        ann = load_annotation("other/results/annotation.gtf.gz")
        return get_tss(ann)

    @pytest.fixture
    def exon(self) -> pl.DataFrame:
        ann = load_annotation("other/results/annotation.gtf.gz")
        return get_exon(ann)

    @pytest.fixture
    def variants_with_exon(self, exon: pl.DataFrame) -> pl.DataFrame:
        V = pl.DataFrame({
            "chrom": ["1", "1", "1"],
            "pos": [65483, 200000, 1471765],
            "ref": ["A", "A", "A"],
            "alt": ["T", "T", "T"],
            "consequence": ["intron_variant", "intergenic_variant", "upstream_gene_variant"],
            "consequence_cre": ["intron_variant", "intergenic_variant", "upstream_gene_variant"],
        })
        return add_exon(V, exon, exon_proximal_dist=100)

    def test_output_columns(self, variants_with_exon: pl.DataFrame, tss: pl.DataFrame) -> None:
        result = add_tss(variants_with_exon, tss)
        assert "tss_dist" in result.columns
        assert "tss_closest_gene_id" in result.columns
        assert "consequence_final" in result.columns

    def test_tss_proximal_overrides_exon_proximal(self, variants_with_exon: pl.DataFrame, tss: pl.DataFrame) -> None:
        # First variant was exon_proximal after add_exon
        assert variants_with_exon.filter(pl.col("pos") == 65483)["consequence_final"][0] == "exon_proximal"

        result = add_tss(variants_with_exon, tss, tss_proximal_dist=1000)
        # Now it should be tss_proximal (intron_variant is in NON_EXONIC and within threshold)
        assert result.filter(pl.col("pos") == 65483)["consequence_final"][0] == "tss_proximal"

    def test_tss_proximal_at_tss(self, variants_with_exon: pl.DataFrame, tss: pl.DataFrame) -> None:
        result = add_tss(variants_with_exon, tss, tss_proximal_dist=1000)
        # Variant at TSS position should be tss_proximal
        assert result.filter(pl.col("pos") == 1471765)["consequence_final"][0] == "tss_proximal"

    def test_distances_match_bioframe(self, variants_with_exon: pl.DataFrame, tss: pl.DataFrame) -> None:
        import bioframe as bf

        result = add_tss(variants_with_exon, tss)

        # Compare with bioframe
        V_pd = variants_with_exon.to_pandas()
        tss_pd = tss.to_pandas()
        V_pd["start"] = V_pd.pos - 1
        V_pd["end"] = V_pd.pos
        result_bf = bf.closest(V_pd, tss_pd)

        assert result["tss_dist"].to_list() == result_bf["distance"].tolist()
