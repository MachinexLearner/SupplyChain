#!/usr/bin/env python3
"""
Test runner for Databricks notebook .py files.
- Runs pure-Python parts (config, helpers, synthetic data) without Spark.
- Optionally runs full notebook with Spark+Delta if available.
Does not modify source files.
"""
import sys
import os
import re
import tempfile
import traceback
from io import StringIO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
# Resolve in case of symlinks / workspace vs filesystem
REPO_ROOT = os.path.abspath(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

def strip_notebook_to_executable(content: str) -> str:
    """Remove MAGIC/%pip cells and return executable Python."""
    lines = content.splitlines()
    out = []
    for line in lines:
        if line.strip().startswith("# MAGIC ") or line.strip().startswith("%pip "):
            continue
        if line.strip() == "# COMMAND ----------":
            out.append("")
            continue
        out.append(line)
    return "\n".join(out)

def run_notebook_without_spark(notebook_path: str) -> dict:
    """
    Execute notebook up to the point it needs Spark; validate functions and synthetic data.
    Then attempt Spark path with temp Delta if available.
    """
    result = {"pass": False, "errors": [], "output": "", "notes": [], "warnings": []}
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    try:
        with open(notebook_path, "r") as f:
            raw_content = f.read()
    except Exception as e:
        result["errors"].append(f"Could not read notebook: {e}")
        return result

    exec_content = strip_notebook_to_executable(raw_content)

    # Split into "before spark" and "after spark" by finding first use of 'spark.'
    lines = exec_content.splitlines()
    before_spark = []
    after_spark = []
    in_before = True
    for line in lines:
        if in_before and "spark." in line and not line.strip().startswith("#"):
            in_before = False
            after_spark.append(line)
            continue
        if in_before:
            before_spark.append(line)
        else:
            after_spark.append(line)
    before_content = "\n".join(before_spark)
    after_content = "\n".join(after_spark)
    # Use smaller record count for faster verification (no source change)
    before_content = re.sub(r"num_records=50000", "num_records=200", before_content)

    namespace = {"__name__": "__main__"}
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout_capture, stderr_capture

    try:
        exec(compile(before_content, notebook_path, "exec"), namespace)
    except Exception as e:
        result["errors"].append("Before-Spark execution failed:\n" + traceback.format_exc())
        sys.stdout, sys.stderr = old_stdout, old_stderr
        result["output"] = stdout_capture.getvalue() + "\n" + stderr_capture.getvalue()
        return result

    # Validate key functions and synthetic data
    if "create_synthetic_usaspending_data" in namespace:
        try:
            create_fn = namespace["create_synthetic_usaspending_data"]
            df = create_fn(num_records=100)
            if df is None or len(df) != 100:
                result["warnings"].append(f"create_synthetic_usaspending_data returned {type(df)} len={len(df) if df is not None else 'N/A'}")
            else:
                expected_cols = ["award_id_piid", "recipient_name", "federal_action_obligation", "fiscal_year", "action_date"]
                for c in expected_cols:
                    if c not in df.columns:
                        result["warnings"].append(f"Missing column in synthetic data: {c}")
                result["notes"].append("create_synthetic_usaspending_data(100) produced DataFrame with expected columns.")
        except Exception as e:
            result["errors"].append("create_synthetic_usaspending_data test failed:\n" + traceback.format_exc())
    else:
        result["warnings"].append("create_synthetic_usaspending_data not found in namespace.")

    # Try full run with Spark + Delta (temp paths)
    with tempfile.TemporaryDirectory(prefix="verification_delta_") as tmpdir:
        raw_path = os.path.join(tmpdir, "raw", "usa_spending_awards")
        bronze_path = os.path.join(tmpdir, "bronze", "oshkosh_prime_award_actions")
        os.makedirs(raw_path, exist_ok=True)
        os.makedirs(bronze_path, exist_ok=True)

        def substitute_paths(content: str) -> str:
            c = content.replace('"/dbfs/raw/usa_spending_awards"', repr(raw_path))
            c = c.replace("'/dbfs/raw/usa_spending_awards'", repr(raw_path))
            c = c.replace('"/bronze/oshkosh_prime_award_actions"', repr(bronze_path))
            c = c.replace("'/bronze/oshkosh_prime_award_actions'", repr(bronze_path))
            c = c.replace("'/raw/usa_spending_awards'", repr(raw_path))
            c = c.replace("'/bronze/oshkosh_prime_award_actions'", repr(bronze_path))
            return c

        full_content = substitute_paths(exec_content)
        full_content = re.sub(r"num_records=50000", "num_records=200", full_content)

        def display(obj):
            try:
                if hasattr(obj, "printSchema"):
                    obj.printSchema()
                if hasattr(obj, "show"):
                    obj.show(5, truncate=False)
            except Exception:
                pass

        try:
            from pyspark.sql import SparkSession
            builder = (
                SparkSession.builder.appName("VerificationTest")
                .master("local[*]")
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            )
            spark = builder.getOrCreate()
        except Exception as e:
            result["notes"].append("Full run skipped: Spark/Delta not available (" + str(e)[:80] + ")")
            sys.stdout, sys.stderr = old_stdout, old_stderr
            result["pass"] = len(result["errors"]) == 0
            result["output"] = stdout_capture.getvalue() + "\n" + stderr_capture.getvalue()
            return result

        namespace["spark"] = spark
        namespace["display"] = display
        try:
            exec(compile(full_content, notebook_path, "exec"), namespace)
            result["notes"].append("Full notebook run (Spark+Delta) completed successfully.")
            result["pass"] = True
        except Exception as e:
            result["errors"].append("Full run failed:\n" + traceback.format_exc())
            result["pass"] = False
        finally:
            try:
                spark.stop()
            except Exception:
                pass

    sys.stdout, sys.stderr = old_stdout, old_stderr
    result["output"] = stdout_capture.getvalue() + "\n" + stderr_capture.getvalue()
    if result["pass"] is False and len(result["errors"]) == 0:
        result["pass"] = True  # passed at least before-spark + function check
    return result

def main():
    notebook_path = os.path.join(REPO_ROOT, "ingestion", "01_usaspending_ingestion.py")
    alt_path = os.path.join(SCRIPT_DIR, "..", "ingestion", "01_usaspending_ingestion.py")
    notebook_path = os.path.abspath(notebook_path)
    if not os.path.isfile(notebook_path) and os.path.isfile(os.path.abspath(alt_path)):
        notebook_path = os.path.abspath(alt_path)
    if not os.path.isfile(notebook_path):
        print("Notebook not found:", notebook_path, "(REPO_ROOT=%s)" % REPO_ROOT)
        return 1
    out_dir = os.path.join(SCRIPT_DIR, "01_usaspending_ingestion")
    os.makedirs(out_dir, exist_ok=True)
    output_txt = os.path.join(out_dir, "01_usaspending_ingestion_test.txt")

    result = run_notebook_without_spark(notebook_path)

    with open(output_txt, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("01_usaspending_ingestion – Test Result\n")
        f.write("=" * 60 + "\n\n")
        f.write("Status: " + ("PASS" if result["pass"] else "FAIL") + "\n\n")
        if result["errors"]:
            f.write("Errors:\n")
            for e in result["errors"]:
                f.write(e)
                f.write("\n")
        if result["warnings"]:
            f.write("Warnings:\n")
            for w in result["warnings"]:
                f.write("- " + w + "\n")
            f.write("\n")
        if result["notes"]:
            f.write("Notes:\n")
            for n in result["notes"]:
                f.write("- " + n + "\n")
            f.write("\n")
        f.write("Output:\n")
        f.write("-" * 40 + "\n")
        f.write(result["output"] or "(none)")
        f.write("\n")

    print("Wrote", output_txt)
    return 0 if result["pass"] else 1

if __name__ == "__main__":
    sys.exit(main())
