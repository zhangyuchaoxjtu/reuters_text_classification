# Helm Chart for Harbor

## 环境

- Kubernetes cluster 1.10.5
- Helm 2.10.0

## 下载

### 增添 Helm 仓库

```bash
helm repo add harbor https://helm.goharbor.io
```

## 配置chart

编辑 `values.yaml` 文件

### 暴露harbor服务

- 由于在配置过程中，没有域名地址，所以我们采用暴露IP的方式来对外提供服务，故选服务类型type选用**NodePort**方式，nodePort的配置信息如下所示：

  ```
  nodePor:
      # The name of NodePort service
      name: harbor
      ports:
        http:
          # The service port Harbor listens on when serving with HTTP`
          port: 80
          # The node port Harbor listens on when serving with HTTP
          nodePort: 30005
        https:
          # The service port Harbor listens on when serving with HTTPS
          port: 443
          # The node port Harbor listens on when serving with HTTPS
          nodePort: 30006
  ```

  

### 配置外部 URL

根据上述配置的nodePort的信息，外部的URL可配置如下：

```
externalURL: http://219.245.186.38:30005
```

### 配置persistence

​	由于集群底层的存储为nfs，所以此部分的storageClass参数配置为"nfs-client"

## 下载 chart

下载Harbor helm chart ，并将其命名为 `my-harbor`:

```bash
helm install --name my-harbor .
```
## 删除chart

删除创建helm chart：`my-harbor` :

```bash
helm delete --purge my-harbor
```
删除和上述chart有关的PVC

```
kubectl delete pvc 
```

## 访问harbor

待harbor相关的pod都是running状态的时候，便可以访问harbor服务

![1602468699926](C:\Users\yczhang\AppData\Roaming\Typora\typora-user-images\1602468699926.png)

访问219.245.186.38:30005 即可访问harbor服务![1602468751170](C:\Users\yczhang\AppData\Roaming\Typora\typora-user-images\1602468751170.png)

账号和密码的设定在上述配置文件values.yaml中完成

## 使用harbor

harbor提供了保存本地镜像的功能，至此便可以将自己的本地的镜像保存至harbor

### 使用docker 登陆进入harbor

```
docker login 219.245.186.38:30005
	admin(values.yaml中设定)
	password(values.yaml中设定)
```

倘若无法登陆，因为docker默认使用的访问协议为https，而我们在设定外部URL的时候使用的是http协议，所以需要在docker内的daemon.json中添加insecure-registry

### 将本地镜像push到harbor

```
docker tag image_Id 219.245.186.38:30005/项目名/镜像tag
示例：
	docker tag 2d87e2e84687 219.245.186.38:30005/library/tf:test
```

push镜像

```
docker push 219.245.186.38:30005/项目名/镜像tag
示例：
	docker push 2d87e2e84687 219.245.186.38:30005/library/tf:test
```

​	若出现retry的情况，并报错overlay/overlay2,则因为镜像版本的底层存储缘故，更改镜像为最新版本即可。