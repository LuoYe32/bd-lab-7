ThisBuild / version      := "0.1.0"
ThisBuild / scalaVersion := "2.12.18"
ThisBuild / organization := "com.datamart"

lazy val root = (project in file("."))
  .settings(
    name := "data-mart",

    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-sql" % "3.5.0" % "provided",

      "com.typesafe" % "config" % "1.4.3",

      "com.squareup.okhttp3" % "okhttp" % "4.12.0",

      "com.fasterxml.jackson.core"   % "jackson-databind"        % "2.15.2",
      "com.fasterxml.jackson.module" %% "jackson-module-scala"   % "2.15.2"
    ),

    assembly / assemblyJarName := "data-mart-assembly.jar",

    assembly / assemblyPackageScala / assembleArtifact := false,

    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", xs @ _*) => MergeStrategy.discard
      case "reference.conf"              => MergeStrategy.concat
      case x =>
        val old = (assembly / assemblyMergeStrategy).value
        old(x)
    }
  )
