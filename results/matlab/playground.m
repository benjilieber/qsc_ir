H_BB = @(x,d) -x.*log2(x./(d-1)) - (1-x).*log2(1-x) ;
H_Bridge = @(x,d) -x.*log2(x) - ((1-x)).*log2((1-x)./d);
p2lambda = @(p,d) p.*(d-1)./d;
p_sift = @(p,d) (((d-1)/d).*p + d-1)./(d+1);


figure
hold on
x = q;
x = dps;
plot(x,qd ./ sift_vec,'ro--','DisplayName','H(Z_A|E)')
plot(x,(qd)./ sift_vec - H_Bridge(q,D) ,'mo--','DisplayName','key rate')
% plot(q,H_3(error./sift_vec /2).*sift_vec,'g-')
% plot(dps,qd - qB,'ro','DisplayName','H(Z_A|E)')
plot(x,H_Bridge(q,D),'go--','DisplayName','error correction')
% plot(q,qB./ sift_vec,'ro','DisplayName','H(A|B)')
ylim([0 1])
legend

% save
S_each = 0
S = 0

prot = Bridge2D; 
D = 2;
q = prot.q;
error = prot.error;
sift_vec = prot.sift_vec;
qd = prot.qd;
qB = prot.qB;
dps = prot.dps;
x = q;
f = figure;
f.Units = 'inches';
f.Position(3:4) = 1.2*[3.5 2.4];
hold on
plot(x,(qd)./ sift_vec - H_Bridge(q,D) ,'r-','LineWidth',1.5)
plot(x,qd ./ sift_vec,'k-.','LineWidth',1.5)
plot(x,H_Bridge(q,D),'b:','LineWidth',1.5)
ylim([0 log2(D+1)+0.1])
xlim([0 0.06])
xlabel("Error Rate, Q",'Interpreter','latex','FontSize', 12) 
ylabel("Information [bits per sifted pulse]",'Interpreter','latex','FontSize', 12)  
box on
legend('boxoff')
legend('Key Rate','$H(A|E)$','$H(A|B)$','Interpreter','latex','FontSize', 12);
legend('Location','best')
f.CurrentAxes.TickLabelInterpreter = 'latex';

if S_each
    saveas(f,'Bridge_2.svg');
    saveas(f,'Bridge_2.png');
end