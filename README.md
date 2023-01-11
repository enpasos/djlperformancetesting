# reproduce bug

testing performance (duration and GPU memory)


## to use the garbageCollection switch - build the feature

```
git clone https://github.com/enpasos/djl.git
cd djl
git checkout gc-orphaned-resources
gradlew build -x test
gradlew publishToMavenLocal
```


## build

```
gradlew build
```
