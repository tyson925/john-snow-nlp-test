package com.blackswan.johnsnow

import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.RegexTokenizer
import com.johnsnowlabs.nlp.annotators.ner.regex.NERRegexApproach
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetectorModel
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, SparkSession}


object NLPTest {

  def createPipeline(data : DataFrame): Unit = {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("body")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetectorModel()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val regexTokenizer = new RegexTokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val nerTagger = new NERRegexApproach()
      .setInputCols(Array("sentence"))
      .setOutputCol("ner")
      .setCorpusPath("./data/dict.txt")

    val finisher = new Finisher()
      .setInputCols("token")
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        regexTokenizer,
        nerTagger,
        finisher
      ))
    pipeline
      .fit(data)
      .transform(data).select("ner")
      .show(10,false)
  }


  def main(args: Array[String]): Unit = {


    val spark = SparkSession.builder().appName("js-demo").master("local[6]").getOrCreate()

    val data = spark.read.json("./data/reuters.json")


    createPipeline(data)
  }
}
