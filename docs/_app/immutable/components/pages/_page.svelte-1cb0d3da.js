import{S as Ce,i as Se,s as Le,l as r,e as ta,J as oa,m as n,h as s,q as e,H as t,n as tt,K as aa,b as Jt,a as u,u as i,p as a,c as f,v as l,L as J,r as Re,M as ia,x as Ge,y as Oe,z as $e,f as Je,t as Ue,A as Ke}from"../../chunks/index-2a41cd7a.js";function ea(g,o,h){const c=g.slice();return c[5]=o[h],c}function ra(g){let o,h;return{c(){o=r("link"),this.h()},l(c){o=n(c,"LINK",{rel:!0,href:!0,as:!0,type:!0,crossorigin:!0}),this.h()},h(){e(o,"rel","preload"),e(o,"href",h=g[5]),e(o,"as","font"),e(o,"type","font/woff2"),e(o,"crossorigin","")},m(c,p){Jt(c,o,p)},p(c,p){p&16&&h!==(h=c[5])&&e(o,"href",h)},d(c){c&&s(o)}}}function la(g){let o,h,c,p,d,b,N,w,y,j,q,P,I,k,A,R;document.title=o=g[0];let W=g[4],E=[];for(let _=0;_<W.length;_+=1)E[_]=ra(ea(g,W,_));return{c(){h=r("meta"),c=r("meta"),p=r("meta"),d=r("meta"),b=r("meta"),N=r("meta"),w=r("meta"),y=r("meta"),j=r("meta"),P=r("meta"),I=r("link"),A=r("link");for(let _=0;_<E.length;_+=1)E[_].c();R=ta(),this.h()},l(_){const v=oa('[data-svelte="svelte-11pdboy"]',document.head);h=n(v,"META",{name:!0,content:!0}),c=n(v,"META",{name:!0,content:!0}),p=n(v,"META",{name:!0,content:!0}),d=n(v,"META",{name:!0,content:!0}),b=n(v,"META",{name:!0,content:!0}),N=n(v,"META",{name:!0,content:!0}),w=n(v,"META",{name:!0,content:!0}),y=n(v,"META",{name:!0,content:!0}),j=n(v,"META",{name:!0,content:!0}),P=n(v,"META",{name:!0,content:!0}),I=n(v,"LINK",{rel:!0,href:!0}),A=n(v,"LINK",{rel:!0,href:!0,integrity:!0,crossorigin:!0,referrerpolicy:!0});for(let m=0;m<E.length;m+=1)E[m].l(v);R=ta(),v.forEach(s),this.h()},h(){e(h,"name","description"),e(h,"content",g[1]),e(c,"name","author"),e(c,"content","Jens E. Pedersen"),e(p,"name","news_keywords"),e(p,"content",g[3]),e(d,"name","twitter:card"),e(d,"content","summary_large_image"),e(b,"name","twitter:site"),e(b,"content","https://jegp.github.io/coordinate-regression"),e(N,"name","twitter:creator"),e(N,"content","@jensegholm"),e(w,"name","twitter:title"),e(w,"content",g[0]),e(y,"name","twitter:description"),e(y,"content",g[1]),e(j,"name","twitter:image:src"),e(j,"content",q=g[2]+"/assets/social-twitter.jpg"),e(P,"name","robots"),e(P,"content","max-image-preview:large"),e(I,"rel","canonical"),e(I,"href",k=g[2]+"/"),e(A,"rel","stylesheet"),e(A,"href","https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.2/css/all.min.css"),e(A,"integrity","sha512-1sCRPdkRXhBV2PBLUdRb4tMg1w2YPf37qatUFeS7zlBy7jJI8Lf4VHwWfZZfpXtYSLy85pkm9GaYVYMfw5BC1A=="),e(A,"crossorigin","anonymous"),e(A,"referrerpolicy","no-referrer")},m(_,v){t(document.head,h),t(document.head,c),t(document.head,p),t(document.head,d),t(document.head,b),t(document.head,N),t(document.head,w),t(document.head,y),t(document.head,j),t(document.head,P),t(document.head,I),t(document.head,A);for(let m=0;m<E.length;m+=1)E[m].m(document.head,null);t(document.head,R)},p(_,[v]){if(v&1&&o!==(o=_[0])&&(document.title=o),v&2&&e(h,"content",_[1]),v&8&&e(p,"content",_[3]),v&1&&e(w,"content",_[0]),v&2&&e(y,"content",_[1]),v&4&&q!==(q=_[2]+"/assets/social-twitter.jpg")&&e(j,"content",q),v&4&&k!==(k=_[2]+"/")&&e(I,"href",k),v&16){W=_[4];let m;for(m=0;m<W.length;m+=1){const T=ea(_,W,m);E[m]?E[m].p(T,v):(E[m]=ra(T),E[m].c(),E[m].m(R.parentNode,R))}for(;m<E.length;m+=1)E[m].d(1);E.length=W.length}},i:tt,o:tt,d(_){s(h),s(c),s(p),s(d),s(b),s(N),s(w),s(y),s(j),s(P),s(I),s(A),aa(E,_),s(R)}}}function ca(g,o,h){let{title:c="Coordinate Regression for event-based vision"}=o,{description:p="Coordinate regression for event-based vision"}=o,{url:d="https://jegp.github.io/coordinate-regression"}=o,{keywords:b="event-based vision, spiking neural networks, machine learning"}=o,{preloadFont:N=[]}=o;return g.$$set=w=>{"title"in w&&h(0,c=w.title),"description"in w&&h(1,p=w.description),"url"in w&&h(2,d=w.url),"keywords"in w&&h(3,b=w.keywords),"preloadFont"in w&&h(4,N=w.preloadFont)},[c,p,d,b,N]}class ha extends Ce{constructor(o){super(),Se(this,o,ca,la,Le,{title:0,description:1,url:2,keywords:3,preloadFont:4})}}function da(g){let o,h,c,p,d,b,N,w,y,j,q,P,I,k,A,R,W,E,_,v,m,T,S,x,V,st,Pt,Y,At,Rt,at,ut,ft,Ye,Xe,Ze,Qe,tr,B,Ut,er,rr,et,nr,Kt,sr,ar,Yt,or,ir,lr,U,cr,Xt,hr,dr,Zt,ur,fr,Qt,pr,mr,vr,te,gr,wr,ee,_r,br,re,Un,kr,z,ot,ne,yr,Er,pt,Tr,se,xr,jr,Ir,F,ae,Nr,zr,mt,qr,Ct,Pr,Ar,Rr,oe,Cr,Sr,vt,gt,Kn,Lr,wt,Yn,Br,ie,Mr,Vr,X,le,Wr,Fr,ce,Hr,Dr,he,_t,Xn,Gr,D,de,Or,$r,bt,Jr,St,Ur,Kr,Yr,ue,kt,Zn,Xr,fe,Zr,Qr,G,pe,tn,en,me,rn,nn,K,ve,sn,an,ge,on,ln,we,cn,hn,_e,dn,un,yt,fn,be,pn,mn,vn,O,ke,gn,wn,ye,Et,Qn,_n,Ee,bn,kn,Te,Lt,ts,yn,Z,xe,En,Tn,je,xn,jn,rt,Ie,In,Nn,Ne,zn,qn,ze,Pn,An,C,qe,Rn,Cn,Tt,Sn,Ln,xt,Bn,Mn,jt,Vn,Wn,It,Fn,Hn,Dn,Gn,nt,Nt,Bt,es,On,zt,Mt,rs,$n,qt,Vt,ns;return{c(){o=r("content"),h=r("div"),c=r("video"),p=r("source"),b=r("source"),w=u(),y=r("div"),j=r("h1"),q=i("Coordinate regression for event-based vision"),P=u(),I=r("div"),k=r("div"),A=i("Authors: "),R=r("a"),W=i("Jens E. Pedersen"),E=i(`, J. P.
					Romero. B, & J. Conradt
					`),_=r("br"),v=i(`
					Contact: `),m=r("i"),T=u(),S=r("a"),x=i("jeped@kth.se"),V=i(`
					- `),st=r("i"),Pt=u(),Y=r("a"),At=i("@jensegholm"),Rt=i(`
					- `),at=r("i"),ut=u(),ft=r("a"),Ye=i("github.com/jegp/coordinate-regression"),Xe=u(),Ze=r("br"),Qe=i(`
				We present a novel method to predict translation-invariant spatial coordinates
				from sparse, event-based vision (EBV) signals using a fully-spiking convolutional
				neural network (SCNN).`),tr=u(),B=r("div"),Ut=r("h2"),er=i("The problem: predicting coordinates with sparse events"),rr=u(),et=r("p"),nr=i(`The video playing in the background illustrates the sparseness of
					event-based vision (EBV) cameras. Most pixels are completely empty
					(~95%), because EBV cameras work by detecting `),Kt=r("b"),sr=i("luminosity changes over time"),ar=i(`. For conventional artificial neural networks this is a challenge.
					Partly because the signal is sparse and
					`),Yt=r("i"),or=i("requires integration over time"),ir=i(`
					(recurrence) to form coherent "pictures" of objects, and partly because
					conventional hardware struggles to keep up with EBV cameras sending more
					than 20M events per second.`),lr=u(),U=r("p"),cr=i(`Our work here focuses on coordinate regression for event-based data,
					aiming to
					`),Xt=r("b"),hr=i("predict center coordinates of geometric shapes"),dr=i(` using
					`),Zt=r("b"),ur=i("spiking neural neural networks"),fr=i(`, designed for
					`),Qt=r("b"),pr=i("neuromorphic hardware"),mr=i("."),vr=u(),te=r("h2"),gr=i("Results: LIF beats ReLU"),wr=u(),ee=r("p"),_r=i(`In our setting, a biologically inspired spiking neural network (SNN)
					with receptive fields (RF) outperform conventional convolutional
					artificial neural network (ANN). The plot below shows the pixel-wise
					error for predictions against unseen validation data for four
					different models.`),br=u(),re=r("img"),kr=u(),z=r("div"),ot=r("div"),ne=r("h2"),yr=i("Why spiking neural networks?"),Er=u(),pt=r("p"),Tr=i(`Spiking neural networks (SNNs) and neuromorphic hardware are inherently
				parallel, asynchronous, low-energy devices that promise accelerated
				compute capacities for event-based machine learning systems. This work
				addresses coordinate regression using an inherently asynchronous and
				parallel spiking architecture that is both compatible with neuromorphic
				hardware, and, we hope, `),se=r("b"),xr=i("a step towards more useful neuromorphic algorithms"),jr=i("."),Ir=u(),F=r("div"),ae=r("h2"),Nr=i("The task: event-based dataset"),zr=u(),mt=r("p"),qr=i(`We construted a dataset of geometric shapes (circles, triangles,
				squares) with center coordinate labels. To provide realistic, structured
				noise, we superimposed the sparse shapes on office-like scene renderings
				from the
				`),Ct=r("a"),Pr=i("NeuroRobotics Platform (NRP)"),Ar=i(`.
				The shapes are sparsely sampled from a Bernouilli distribution (p=0.8)
				and are moving around with a brownian motion. The task is for any
				network to recognize the shapes and predict their center as accurately
				as possible.`),Rr=u(),oe=r("p"),Cr=i(`We chose the geometric shapes as a means to control parameters such as
				shape velocity, event density, and shape complexity. For instance, the
				current shapes have a radius of 80 pixels, which is too large for any
				single kernel to learn. The convolutional layer is, therefore, forced to
				specialize on partial features to correctly solve the task.`),Sr=u(),vt=r("center"),gt=r("img"),Lr=u(),wt=r("img"),Br=u(),ie=r("p"),Mr=i("In total, the dataset contains 2000 videos of 60 frames each (resembling 1ms of events) with a resolution of 640x480."),Vr=u(),X=r("div"),le=r("h2"),Wr=i("Neural network architecture"),Fr=u(),ce=r("p"),Hr=i(`We use a classical convolutional model consisting of two convolutional
				layers, followed by an inverse convolution for a slight upsampling. We
				implemented a differentiable coordinate transform (DCT) layer to
				transform the pixel-wise activations into a 2-dimensional coordinate.
				Each convolution is interspersed with non-linear activations, batch
				normalization and dropout (p=0.1).`),Dr=u(),he=r("center"),_t=r("img"),Gr=u(),D=r("div"),de=r("h2"),Or=i("Translation-invariant receptive fields"),$r=u(),bt=r("p"),Jr=i(`Similar to conventional convolutional systems, we can define receptive
				field kernels for spiking neural networks. Importantly, we wish to
				retain translation-invariant properties to capture the moving shapes
				over time, as neatly illustrates in the work by `),St=r("a"),Ur=i("[Lindeberg 2021]"),Kr=i("."),Yr=u(),ue=r("center"),kt=r("img"),Xr=u(),fe=r("p"),Zr=i(`Preconfiguring the receptive field kernels for the spiking neural
				networks significantly reduces training time, memory consumption, and
				ability for the network to generalize, as seen in the loss curves above.`),Qr=u(),G=r("div"),pe=r("h2"),tn=i("Training and validation"),en=u(),me=r("p"),rn=i(`We constructed and trained four independent networks over the same
				architecture`),nn=u(),K=r("ol"),ve=r("li"),sn=i(`a non-spiking network where ReLU units constitute the
				nonlinearities (ANN)`),an=u(),ge=r("li"),on=i("a non-spiking network with custom receptive field kernels (ANN-RF)"),ln=u(),we=r("li"),cn=i(`a spiking version, with three leaky integrate-and-fire (LIF)
				nonlinearities feeding a final, non-spiking leaky integrator (SNN)`),hn=u(),_e=r("li"),dn=i(`a
				spiking version, resembling (3), but where custom receptive field
				kernels promote translation-invariant feature recognition (SNN-RF)`),un=u(),yt=r("p"),fn=i(`The networks were trained with backpropagation-through-time using a regular
				l`),be=r("sup"),pn=i("2"),mn=i(`-loss via the differentiable coordinate transform (DCT)
				method (presented futher below). The models are tested on unseen
				validation data (20% of the total training data).`),vn=u(),O=r("div"),ke=r("h2"),gn=i("Prediction errors and performance"),wn=u(),ye=r("center"),Et=r("img"),_n=u(),Ee=r("p"),bn=i(`The receptive field version of the SNN outperforms even the artificial
				neural network. If we further explore the prediction errors of the
				models (sampled over the entire validation dataset), the performance
				benefit becomes clearer: the predicted coordinates from the receptive
				field model is significantly closer to the actual, labelled coordinates.`),kn=u(),Te=r("center"),Lt=r("img"),yn=u(),Z=r("div"),xe=r("h2"),En=i("Future work"),Tn=u(),je=r("p"),xn=i(`This is still work in progress and more work is needed to generalize the results.
				However, there are already a few extensions that would immediately be interesting to explore`),jn=u(),rt=r("ol"),Ie=r("li"),In=i(`The gaussian structure of the output predictions can be exploited to
					further increase prediction accuracy.`),Nn=u(),Ne=r("li"),zn=i(`The current shapes are quite dense (Bernouilli p=0.8), such that the artificial networks
					are able to converge to the shapes in the given frames. We wish to
					explore the sparseness of the shapes (lower the Bernouilli
					distribution of the shapes) while exploring the temporal process of
					recurrent artificial (non-spiking) networks.`),qn=u(),ze=r("li"),Pn=i(`We currently focus on translation-invariance. We wish to extend our
					method to more complex shapes that require both scale- and
					rotation-invariance.`),An=u(),C=r("div"),qe=r("h2"),Rn=i("Acknowledgements"),Cn=i(`
			Thank you to all the people in the
			`),Tt=r("a"),Sn=i("Neurocomputing Systems Lab"),Ln=i(`.
			at
			`),xt=r("a"),Bn=i("KTH Royal Institute of Technology"),Mn=i(`, where this
			work was done. We graciously acknowledge the funding we received from the
			`),jt=r("a"),Vn=i("Human Brain Project"),Wn=i(`. We
			also owe a debt of gratitude to the
			`),It=r("a"),Fn=i("Copenhagen AI Pioneer Centre"),Hn=i(`.
			`),Dn=r("br"),Gn=u(),nt=r("center"),Nt=r("a"),Bt=r("img"),On=u(),zt=r("a"),Mt=r("img"),$n=u(),qt=r("a"),Vt=r("img"),this.h()},l(Pe){o=n(Pe,"CONTENT",{class:!0});var Wt=a(o);h=n(Wt,"DIV",{class:!0});var Be=a(h);c=n(Be,"VIDEO",{class:!0});var Jn=a(c);p=n(Jn,"SOURCE",{src:!0,type:!0}),b=n(Jn,"SOURCE",{src:!0,type:!0}),Jn.forEach(s),w=f(Be),y=n(Be,"DIV",{class:!0});var Ft=a(y);j=n(Ft,"H1",{class:!0});var ss=a(j);q=l(ss,"Coordinate regression for event-based vision"),ss.forEach(s),P=f(Ft),I=n(Ft,"DIV",{class:!0});var Ae=a(I);k=n(Ae,"DIV",{class:!0});var L=a(k);A=l(L,"Authors: "),R=n(L,"A",{href:!0,class:!0});var as=a(R);W=l(as,"Jens E. Pedersen"),as.forEach(s),E=l(L,`, J. P.
					Romero. B, & J. Conradt
					`),_=n(L,"BR",{}),v=l(L,`
					Contact: `),m=n(L,"I",{class:!0}),a(m).forEach(s),T=f(L),S=n(L,"A",{href:!0,class:!0});var os=a(S);x=l(os,"jeped@kth.se"),os.forEach(s),V=l(L,`
					- `),st=n(L,"I",{class:!0}),a(st).forEach(s),Pt=f(L),Y=n(L,"A",{href:!0,class:!0});var is=a(Y);At=l(is,"@jensegholm"),is.forEach(s),Rt=l(L,`
					- `),at=n(L,"I",{class:!0}),a(at).forEach(s),ut=f(L),ft=n(L,"A",{href:!0,class:!0});var ls=a(ft);Ye=l(ls,"github.com/jegp/coordinate-regression"),ls.forEach(s),L.forEach(s),Xe=f(Ae),Ze=n(Ae,"BR",{}),Qe=l(Ae,`
				We present a novel method to predict translation-invariant spatial coordinates
				from sparse, event-based vision (EBV) signals using a fully-spiking convolutional
				neural network (SCNN).`),Ae.forEach(s),tr=f(Ft),B=n(Ft,"DIV",{class:!0});var $=a(B);Ut=n($,"H2",{});var cs=a(Ut);er=l(cs,"The problem: predicting coordinates with sparse events"),cs.forEach(s),rr=f($),et=n($,"P",{});var Ht=a(et);nr=l(Ht,`The video playing in the background illustrates the sparseness of
					event-based vision (EBV) cameras. Most pixels are completely empty
					(~95%), because EBV cameras work by detecting `),Kt=n(Ht,"B",{});var hs=a(Kt);sr=l(hs,"luminosity changes over time"),hs.forEach(s),ar=l(Ht,`. For conventional artificial neural networks this is a challenge.
					Partly because the signal is sparse and
					`),Yt=n(Ht,"I",{});var ds=a(Yt);or=l(ds,"requires integration over time"),ds.forEach(s),ir=l(Ht,`
					(recurrence) to form coherent "pictures" of objects, and partly because
					conventional hardware struggles to keep up with EBV cameras sending more
					than 20M events per second.`),Ht.forEach(s),lr=f($),U=n($,"P",{});var it=a(U);cr=l(it,`Our work here focuses on coordinate regression for event-based data,
					aiming to
					`),Xt=n(it,"B",{});var us=a(Xt);hr=l(us,"predict center coordinates of geometric shapes"),us.forEach(s),dr=l(it,` using
					`),Zt=n(it,"B",{});var fs=a(Zt);ur=l(fs,"spiking neural neural networks"),fs.forEach(s),fr=l(it,`, designed for
					`),Qt=n(it,"B",{});var ps=a(Qt);pr=l(ps,"neuromorphic hardware"),ps.forEach(s),mr=l(it,"."),it.forEach(s),vr=f($),te=n($,"H2",{});var ms=a(te);gr=l(ms,"Results: LIF beats ReLU"),ms.forEach(s),wr=f($),ee=n($,"P",{});var vs=a(ee);_r=l(vs,`In our setting, a biologically inspired spiking neural network (SNN)
					with receptive fields (RF) outperform conventional convolutional
					artificial neural network (ANN). The plot below shows the pixel-wise
					error for predictions against unseen validation data for four
					different models.`),vs.forEach(s),br=f($),re=n($,"IMG",{src:!0}),$.forEach(s),Ft.forEach(s),Be.forEach(s),kr=f(Wt),z=n(Wt,"DIV",{class:!0});var M=a(z);ot=n(M,"DIV",{class:!0});var Me=a(ot);ne=n(Me,"H2",{});var gs=a(ne);yr=l(gs,"Why spiking neural networks?"),gs.forEach(s),Er=f(Me),pt=n(Me,"P",{});var Ve=a(pt);Tr=l(Ve,`Spiking neural networks (SNNs) and neuromorphic hardware are inherently
				parallel, asynchronous, low-energy devices that promise accelerated
				compute capacities for event-based machine learning systems. This work
				addresses coordinate regression using an inherently asynchronous and
				parallel spiking architecture that is both compatible with neuromorphic
				hardware, and, we hope, `),se=n(Ve,"B",{});var ws=a(se);xr=l(ws,"a step towards more useful neuromorphic algorithms"),ws.forEach(s),jr=l(Ve,"."),Ve.forEach(s),Me.forEach(s),Ir=f(M),F=n(M,"DIV",{class:!0});var Q=a(F);ae=n(Q,"H2",{});var _s=a(ae);Nr=l(_s,"The task: event-based dataset"),_s.forEach(s),zr=f(Q),mt=n(Q,"P",{});var We=a(mt);qr=l(We,`We construted a dataset of geometric shapes (circles, triangles,
				squares) with center coordinate labels. To provide realistic, structured
				noise, we superimposed the sparse shapes on office-like scene renderings
				from the
				`),Ct=n(We,"A",{href:!0});var bs=a(Ct);Pr=l(bs,"NeuroRobotics Platform (NRP)"),bs.forEach(s),Ar=l(We,`.
				The shapes are sparsely sampled from a Bernouilli distribution (p=0.8)
				and are moving around with a brownian motion. The task is for any
				network to recognize the shapes and predict their center as accurately
				as possible.`),We.forEach(s),Rr=f(Q),oe=n(Q,"P",{});var ks=a(oe);Cr=l(ks,`We chose the geometric shapes as a means to control parameters such as
				shape velocity, event density, and shape complexity. For instance, the
				current shapes have a radius of 80 pixels, which is too large for any
				single kernel to learn. The convolutional layer is, therefore, forced to
				specialize on partial features to correctly solve the task.`),ks.forEach(s),Sr=f(Q),vt=n(Q,"CENTER",{});var Fe=a(vt);gt=n(Fe,"IMG",{src:!0,style:!0,class:!0}),Lr=f(Fe),wt=n(Fe,"IMG",{src:!0,style:!0,class:!0}),Fe.forEach(s),Br=f(Q),ie=n(Q,"P",{});var ys=a(ie);Mr=l(ys,"In total, the dataset contains 2000 videos of 60 frames each (resembling 1ms of events) with a resolution of 640x480."),ys.forEach(s),Q.forEach(s),Vr=f(M),X=n(M,"DIV",{class:!0});var Dt=a(X);le=n(Dt,"H2",{});var Es=a(le);Wr=l(Es,"Neural network architecture"),Es.forEach(s),Fr=f(Dt),ce=n(Dt,"P",{});var Ts=a(ce);Hr=l(Ts,`We use a classical convolutional model consisting of two convolutional
				layers, followed by an inverse convolution for a slight upsampling. We
				implemented a differentiable coordinate transform (DCT) layer to
				transform the pixel-wise activations into a 2-dimensional coordinate.
				Each convolution is interspersed with non-linear activations, batch
				normalization and dropout (p=0.1).`),Ts.forEach(s),Dr=f(Dt),he=n(Dt,"CENTER",{});var xs=a(he);_t=n(xs,"IMG",{src:!0,style:!0,class:!0}),xs.forEach(s),Dt.forEach(s),Gr=f(M),D=n(M,"DIV",{class:!0});var lt=a(D);de=n(lt,"H2",{});var js=a(de);Or=l(js,"Translation-invariant receptive fields"),js.forEach(s),$r=f(lt),bt=n(lt,"P",{});var He=a(bt);Jr=l(He,`Similar to conventional convolutional systems, we can define receptive
				field kernels for spiking neural networks. Importantly, we wish to
				retain translation-invariant properties to capture the moving shapes
				over time, as neatly illustrates in the work by `),St=n(He,"A",{href:!0});var Is=a(St);Ur=l(Is,"[Lindeberg 2021]"),Is.forEach(s),Kr=l(He,"."),He.forEach(s),Yr=f(lt),ue=n(lt,"CENTER",{});var Ns=a(ue);kt=n(Ns,"IMG",{src:!0,style:!0,class:!0}),Ns.forEach(s),Xr=f(lt),fe=n(lt,"P",{});var zs=a(fe);Zr=l(zs,`Preconfiguring the receptive field kernels for the spiking neural
				networks significantly reduces training time, memory consumption, and
				ability for the network to generalize, as seen in the loss curves above.`),zs.forEach(s),lt.forEach(s),Qr=f(M),G=n(M,"DIV",{class:!0});var ct=a(G);pe=n(ct,"H2",{});var qs=a(pe);tn=l(qs,"Training and validation"),qs.forEach(s),en=f(ct),me=n(ct,"P",{});var Ps=a(me);rn=l(Ps,`We constructed and trained four independent networks over the same
				architecture`),Ps.forEach(s),nn=f(ct),K=n(ct,"OL",{});var ht=a(K);ve=n(ht,"LI",{});var As=a(ve);sn=l(As,`a non-spiking network where ReLU units constitute the
				nonlinearities (ANN)`),As.forEach(s),an=f(ht),ge=n(ht,"LI",{});var Rs=a(ge);on=l(Rs,"a non-spiking network with custom receptive field kernels (ANN-RF)"),Rs.forEach(s),ln=f(ht),we=n(ht,"LI",{});var Cs=a(we);cn=l(Cs,`a spiking version, with three leaky integrate-and-fire (LIF)
				nonlinearities feeding a final, non-spiking leaky integrator (SNN)`),Cs.forEach(s),hn=f(ht),_e=n(ht,"LI",{});var Ss=a(_e);dn=l(Ss,`a
				spiking version, resembling (3), but where custom receptive field
				kernels promote translation-invariant feature recognition (SNN-RF)`),Ss.forEach(s),ht.forEach(s),un=f(ct),yt=n(ct,"P",{});var De=a(yt);fn=l(De,`The networks were trained with backpropagation-through-time using a regular
				l`),be=n(De,"SUP",{});var Ls=a(be);pn=l(Ls,"2"),Ls.forEach(s),mn=l(De,`-loss via the differentiable coordinate transform (DCT)
				method (presented futher below). The models are tested on unseen
				validation data (20% of the total training data).`),De.forEach(s),ct.forEach(s),vn=f(M),O=n(M,"DIV",{class:!0});var dt=a(O);ke=n(dt,"H2",{});var Bs=a(ke);gn=l(Bs,"Prediction errors and performance"),Bs.forEach(s),wn=f(dt),ye=n(dt,"CENTER",{});var Ms=a(ye);Et=n(Ms,"IMG",{src:!0,style:!0,class:!0}),Ms.forEach(s),_n=f(dt),Ee=n(dt,"P",{});var Vs=a(Ee);bn=l(Vs,`The receptive field version of the SNN outperforms even the artificial
				neural network. If we further explore the prediction errors of the
				models (sampled over the entire validation dataset), the performance
				benefit becomes clearer: the predicted coordinates from the receptive
				field model is significantly closer to the actual, labelled coordinates.`),Vs.forEach(s),kn=f(dt),Te=n(dt,"CENTER",{});var Ws=a(Te);Lt=n(Ws,"IMG",{src:!0,class:!0}),Ws.forEach(s),dt.forEach(s),yn=f(M),Z=n(M,"DIV",{class:!0});var Gt=a(Z);xe=n(Gt,"H2",{});var Fs=a(xe);En=l(Fs,"Future work"),Fs.forEach(s),Tn=f(Gt),je=n(Gt,"P",{});var Hs=a(je);xn=l(Hs,`This is still work in progress and more work is needed to generalize the results.
				However, there are already a few extensions that would immediately be interesting to explore`),Hs.forEach(s),jn=f(Gt),rt=n(Gt,"OL",{});var Ot=a(rt);Ie=n(Ot,"LI",{});var Ds=a(Ie);In=l(Ds,`The gaussian structure of the output predictions can be exploited to
					further increase prediction accuracy.`),Ds.forEach(s),Nn=f(Ot),Ne=n(Ot,"LI",{});var Gs=a(Ne);zn=l(Gs,`The current shapes are quite dense (Bernouilli p=0.8), such that the artificial networks
					are able to converge to the shapes in the given frames. We wish to
					explore the sparseness of the shapes (lower the Bernouilli
					distribution of the shapes) while exploring the temporal process of
					recurrent artificial (non-spiking) networks.`),Gs.forEach(s),qn=f(Ot),ze=n(Ot,"LI",{});var Os=a(ze);Pn=l(Os,`We currently focus on translation-invariance. We wish to extend our
					method to more complex shapes that require both scale- and
					rotation-invariance.`),Os.forEach(s),Ot.forEach(s),Gt.forEach(s),An=f(M),C=n(M,"DIV",{class:!0});var H=a(C);qe=n(H,"H2",{});var $s=a(qe);Rn=l($s,"Acknowledgements"),$s.forEach(s),Cn=l(H,`
			Thank you to all the people in the
			`),Tt=n(H,"A",{href:!0,class:!0});var Js=a(Tt);Sn=l(Js,"Neurocomputing Systems Lab"),Js.forEach(s),Ln=l(H,`.
			at
			`),xt=n(H,"A",{href:!0,class:!0});var Us=a(xt);Bn=l(Us,"KTH Royal Institute of Technology"),Us.forEach(s),Mn=l(H,`, where this
			work was done. We graciously acknowledge the funding we received from the
			`),jt=n(H,"A",{href:!0,class:!0});var Ks=a(jt);Vn=l(Ks,"Human Brain Project"),Ks.forEach(s),Wn=l(H,`. We
			also owe a debt of gratitude to the
			`),It=n(H,"A",{href:!0,class:!0});var Ys=a(It);Fn=l(Ys,"Copenhagen AI Pioneer Centre"),Ys.forEach(s),Hn=l(H,`.
			`),Dn=n(H,"BR",{}),Gn=f(H),nt=n(H,"CENTER",{});var $t=a(nt);Nt=n($t,"A",{class:!0,href:!0});var Xs=a(Nt);Bt=n(Xs,"IMG",{src:!0,class:!0}),Xs.forEach(s),On=f($t),zt=n($t,"A",{class:!0,href:!0});var Zs=a(zt);Mt=n(Zs,"IMG",{src:!0,class:!0}),Zs.forEach(s),$n=f($t),qt=n($t,"A",{class:!0,href:!0});var Qs=a(qt);Vt=n(Qs,"IMG",{src:!0,class:!0}),Qs.forEach(s),$t.forEach(s),H.forEach(s),M.forEach(s),Wt.forEach(s),this.h()},h(){J(p.src,d="2209_event_video.mp4")||e(p,"src",d),e(p,"type","video/mp4"),J(b.src,N="2209_event_video.webm")||e(b,"src",N),e(b,"type","video/webm"),e(c,"class","background svelte-zqxcjm"),c.autoplay=!0,c.loop=!0,c.muted=!0,e(j,"class","svelte-zqxcjm"),e(R,"href","https://jepedersen.dk"),e(R,"class","svelte-zqxcjm"),e(m,"class","fa fa-envelope"),e(S,"href","mailto:jeped@kth.se"),e(S,"class","svelte-zqxcjm"),e(st,"class","fab fa-twitter"),e(Y,"href","https://twitter.com/jensegholm"),e(Y,"class","svelte-zqxcjm"),e(at,"class","fab fa-github"),e(ft,"href","https://github.com/jegp/coordinate-regression"),e(ft,"class","svelte-zqxcjm"),e(k,"class","author svelte-zqxcjm"),e(I,"class","box svelte-zqxcjm"),J(re.src,Un="pred_horiz.png")||e(re,"src",Un),e(B,"class","box svelte-zqxcjm"),e(y,"class","hover svelte-zqxcjm"),e(h,"class","head svelte-zqxcjm"),e(ot,"class","section svelte-zqxcjm"),e(Ct,"href","https://neurorobotics.net/"),J(gt.src,Kn="fig1_color.png")||e(gt,"src",Kn),Re(gt,"width","30%"),e(gt,"class","svelte-zqxcjm"),J(wt.src,Yn="fig1_circle.png")||e(wt,"src",Yn),Re(wt,"width","30%"),e(wt,"class","svelte-zqxcjm"),e(F,"class","section svelte-zqxcjm"),J(_t.src,Xn="net.png")||e(_t,"src",Xn),Re(_t,"width","80%"),e(_t,"class","svelte-zqxcjm"),e(X,"class","section svelte-zqxcjm"),e(St,"href","https://www.sciencedirect.com/science/article/pii/S2405844021000025"),J(kt.src,Zn="fig_rf.png")||e(kt,"src",Zn),Re(kt,"width","60%"),e(kt,"class","svelte-zqxcjm"),e(D,"class","section svelte-zqxcjm"),e(G,"class","section svelte-zqxcjm"),J(Et.src,Qn="fig_loss2.png")||e(Et,"src",Qn),Re(Et,"width","80%"),e(Et,"class","svelte-zqxcjm"),J(Lt.src,ts="pred_horiz.png")||e(Lt,"src",ts),e(Lt,"class","svelte-zqxcjm"),e(O,"class","section svelte-zqxcjm"),e(Z,"class","section svelte-zqxcjm"),e(Tt,"href","https://neurocomputing.systems"),e(Tt,"class","svelte-zqxcjm"),e(xt,"href","https://kth.se"),e(xt,"class","svelte-zqxcjm"),e(jt,"href","https://www.humanbrainproject.eu/"),e(jt,"class","svelte-zqxcjm"),e(It,"href","https://www.aicentre.dk/"),e(It,"class","svelte-zqxcjm"),J(Bt.src,es="ncs.png")||e(Bt,"src",es),e(Bt,"class","svelte-zqxcjm"),e(Nt,"class","borderless svelte-zqxcjm"),e(Nt,"href","https://neurocomputing.systems"),J(Mt.src,rs="hbp.png")||e(Mt,"src",rs),e(Mt,"class","svelte-zqxcjm"),e(zt,"class","borderless svelte-zqxcjm"),e(zt,"href","https://humanbrainproject.eu"),J(Vt.src,ns="kth.png")||e(Vt,"src",ns),e(Vt,"class","svelte-zqxcjm"),e(qt,"class","borderless svelte-zqxcjm"),e(qt,"href","https://kth.se"),e(C,"class","box author svelte-zqxcjm"),e(z,"class","block svelte-zqxcjm"),e(o,"class","svelte-zqxcjm")},m(Pe,Wt){Jt(Pe,o,Wt),t(o,h),t(h,c),t(c,p),t(c,b),t(h,w),t(h,y),t(y,j),t(j,q),t(y,P),t(y,I),t(I,k),t(k,A),t(k,R),t(R,W),t(k,E),t(k,_),t(k,v),t(k,m),t(k,T),t(k,S),t(S,x),t(k,V),t(k,st),t(k,Pt),t(k,Y),t(Y,At),t(k,Rt),t(k,at),t(k,ut),t(k,ft),t(ft,Ye),t(I,Xe),t(I,Ze),t(I,Qe),t(y,tr),t(y,B),t(B,Ut),t(Ut,er),t(B,rr),t(B,et),t(et,nr),t(et,Kt),t(Kt,sr),t(et,ar),t(et,Yt),t(Yt,or),t(et,ir),t(B,lr),t(B,U),t(U,cr),t(U,Xt),t(Xt,hr),t(U,dr),t(U,Zt),t(Zt,ur),t(U,fr),t(U,Qt),t(Qt,pr),t(U,mr),t(B,vr),t(B,te),t(te,gr),t(B,wr),t(B,ee),t(ee,_r),t(B,br),t(B,re),t(o,kr),t(o,z),t(z,ot),t(ot,ne),t(ne,yr),t(ot,Er),t(ot,pt),t(pt,Tr),t(pt,se),t(se,xr),t(pt,jr),t(z,Ir),t(z,F),t(F,ae),t(ae,Nr),t(F,zr),t(F,mt),t(mt,qr),t(mt,Ct),t(Ct,Pr),t(mt,Ar),t(F,Rr),t(F,oe),t(oe,Cr),t(F,Sr),t(F,vt),t(vt,gt),t(vt,Lr),t(vt,wt),t(F,Br),t(F,ie),t(ie,Mr),t(z,Vr),t(z,X),t(X,le),t(le,Wr),t(X,Fr),t(X,ce),t(ce,Hr),t(X,Dr),t(X,he),t(he,_t),t(z,Gr),t(z,D),t(D,de),t(de,Or),t(D,$r),t(D,bt),t(bt,Jr),t(bt,St),t(St,Ur),t(bt,Kr),t(D,Yr),t(D,ue),t(ue,kt),t(D,Xr),t(D,fe),t(fe,Zr),t(z,Qr),t(z,G),t(G,pe),t(pe,tn),t(G,en),t(G,me),t(me,rn),t(G,nn),t(G,K),t(K,ve),t(ve,sn),t(K,an),t(K,ge),t(ge,on),t(K,ln),t(K,we),t(we,cn),t(K,hn),t(K,_e),t(_e,dn),t(G,un),t(G,yt),t(yt,fn),t(yt,be),t(be,pn),t(yt,mn),t(z,vn),t(z,O),t(O,ke),t(ke,gn),t(O,wn),t(O,ye),t(ye,Et),t(O,_n),t(O,Ee),t(Ee,bn),t(O,kn),t(O,Te),t(Te,Lt),t(z,yn),t(z,Z),t(Z,xe),t(xe,En),t(Z,Tn),t(Z,je),t(je,xn),t(Z,jn),t(Z,rt),t(rt,Ie),t(Ie,In),t(rt,Nn),t(rt,Ne),t(Ne,zn),t(rt,qn),t(rt,ze),t(ze,Pn),t(z,An),t(z,C),t(C,qe),t(qe,Rn),t(C,Cn),t(C,Tt),t(Tt,Sn),t(C,Ln),t(C,xt),t(xt,Bn),t(C,Mn),t(C,jt),t(jt,Vn),t(C,Wn),t(C,It),t(It,Fn),t(C,Hn),t(C,Dn),t(C,Gn),t(C,nt),t(nt,Nt),t(Nt,Bt),t(nt,On),t(nt,zt),t(zt,Mt),t(nt,$n),t(nt,qt),t(qt,Vt)},p:tt,i:tt,o:tt,d(Pe){Pe&&s(o)}}}class ua extends Ce{constructor(o){super(),Se(this,o,null,da,Le,{})}}function na(g,o,h){const c=g.slice();return c[1]=o[h],c}function sa(g){let o,h,c,p,d,b=g[1].name.toUpperCase()+"",N,w;return{c(){o=r("li"),h=r("a"),c=r("span"),p=r("i"),d=u(),N=i(b),w=u(),this.h()},l(y){o=n(y,"LI",{class:!0});var j=a(o);h=n(j,"A",{href:!0,class:!0});var q=a(h);c=n(q,"SPAN",{class:!0});var P=a(c);p=n(P,"I",{class:!0}),a(p).forEach(s),d=f(P),N=l(P,b),P.forEach(s),q.forEach(s),w=f(j),j.forEach(s),this.h()},h(){e(p,"class",ia(g[1].icon)+" svelte-1pc98p8"),e(c,"class","svelte-1pc98p8"),e(h,"href",g[1].url),e(h,"class","svelte-1pc98p8"),e(o,"class","svelte-1pc98p8")},m(y,j){Jt(y,o,j),t(o,h),t(h,c),t(c,p),t(c,d),t(c,N),t(o,w)},p:tt,d(y){y&&s(o)}}}function fa(g){let o,h,c,p,d,b,N,w,y,j,q,P,I,k,A,R,W,E,_,v=g[0],m=[];for(let T=0;T<v.length;T+=1)m[T]=sa(na(g,v,T));return{c(){o=r("footer"),h=r("section"),c=r("p"),p=i("This website was developed by "),d=r("a"),b=i("Jens Egholm Pedersen"),N=i(`
			who is studying at the
			`),w=r("a"),y=i("Neurocomputing Systems lab"),j=i(`
			at the `),q=r("a"),P=i("KTH Royal Institute of Technology"),I=i(` in Stockholm,
			Sweden.`),k=u(),A=r("p"),R=i("The work is licensed under LGPLv3."),W=u(),E=r("section"),_=r("ul");for(let T=0;T<m.length;T+=1)m[T].c();this.h()},l(T){o=n(T,"FOOTER",{class:!0});var S=a(o);h=n(S,"SECTION",{class:!0});var x=a(h);c=n(x,"P",{});var V=a(c);p=l(V,"This website was developed by "),d=n(V,"A",{href:!0,class:!0});var st=a(d);b=l(st,"Jens Egholm Pedersen"),st.forEach(s),N=l(V,`
			who is studying at the
			`),w=n(V,"A",{href:!0,class:!0});var Pt=a(w);y=l(Pt,"Neurocomputing Systems lab"),Pt.forEach(s),j=l(V,`
			at the `),q=n(V,"A",{href:!0,class:!0});var Y=a(q);P=l(Y,"KTH Royal Institute of Technology"),Y.forEach(s),I=l(V,` in Stockholm,
			Sweden.`),V.forEach(s),k=f(x),A=n(x,"P",{});var At=a(A);R=l(At,"The work is licensed under LGPLv3."),At.forEach(s),x.forEach(s),W=f(S),E=n(S,"SECTION",{class:!0});var Rt=a(E);_=n(Rt,"UL",{class:!0});var at=a(_);for(let ut=0;ut<m.length;ut+=1)m[ut].l(at);at.forEach(s),Rt.forEach(s),S.forEach(s),this.h()},h(){e(d,"href","https://jepedersen.dk"),e(d,"class","svelte-1pc98p8"),e(w,"href","https://neurocomputing.systems"),e(w,"class","svelte-1pc98p8"),e(q,"href","https://kth.se"),e(q,"class","svelte-1pc98p8"),e(h,"class","about svelte-1pc98p8"),e(_,"class","svelte-1pc98p8"),e(E,"class","links svelte-1pc98p8"),e(o,"class","svelte-1pc98p8")},m(T,S){Jt(T,o,S),t(o,h),t(h,c),t(c,p),t(c,d),t(d,b),t(c,N),t(c,w),t(w,y),t(c,j),t(c,q),t(q,P),t(c,I),t(h,k),t(h,A),t(A,R),t(o,W),t(o,E),t(E,_);for(let x=0;x<m.length;x+=1)m[x].m(_,null)},p(T,[S]){if(S&1){v=T[0];let x;for(x=0;x<v.length;x+=1){const V=na(T,v,x);m[x]?m[x].p(V,S):(m[x]=sa(V),m[x].c(),m[x].m(_,null))}for(;x<m.length;x+=1)m[x].d(1);m.length=v.length}},i:tt,o:tt,d(T){T&&s(o),aa(m,T)}}}function pa(g){return[[{name:"about",url:"https://jepedersen.dk/about",icon:"fa-solid fa-house"},{name:"twitter",url:"https://twitter.com/jensegholm/",icon:"fab fa-twitter"},{name:"GitHub",url:"https://github.com/jegp/coordinate-regression/",icon:"fab fa-github"}]]}class ma extends Ce{constructor(o){super(),Se(this,o,pa,fa,Le,{})}}function va(g){let o,h,c,p;return o=new ua({}),c=new ma({}),{c(){Ge(o.$$.fragment),h=u(),Ge(c.$$.fragment)},l(d){Oe(o.$$.fragment,d),h=f(d),Oe(c.$$.fragment,d)},m(d,b){$e(o,d,b),Jt(d,h,b),$e(c,d,b),p=!0},p:tt,i(d){p||(Je(o.$$.fragment,d),Je(c.$$.fragment,d),p=!0)},o(d){Ue(o.$$.fragment,d),Ue(c.$$.fragment,d),p=!1},d(d){Ke(o,d),d&&s(h),Ke(c,d)}}}class ga extends Ce{constructor(o){super(),Se(this,o,null,va,Le,{})}}function wa(){console.log("--- --- --- --- --- ---"),console.log("svelte-starter: 4.0.1"),console.log("build: 2023-03-17-12:56"),console.log("--- --- --- --- --- ---")}function _a(g){let o,h,c,p;return o=new ha({}),c=new ga({}),{c(){Ge(o.$$.fragment),h=u(),Ge(c.$$.fragment)},l(d){Oe(o.$$.fragment,d),h=f(d),Oe(c.$$.fragment,d)},m(d,b){$e(o,d,b),Jt(d,h,b),$e(c,d,b),p=!0},p:tt,i(d){p||(Je(o.$$.fragment,d),Je(c.$$.fragment,d),p=!0)},o(d){Ue(o.$$.fragment,d),Ue(c.$$.fragment,d),p=!1},d(d){Ke(o,d),d&&s(h),Ke(c,d)}}}function ba(g,o,h){let{data:c}=o;return wa(),g.$$set=p=>{"data"in p&&h(0,c=p.data)},[c]}class ya extends Ce{constructor(o){super(),Se(this,o,ba,_a,Le,{data:0})}}export{ya as default};
